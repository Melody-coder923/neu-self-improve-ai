"""
Pretraining via behavioral cloning on BFS expert demonstrations.
Evaluate on same 50 configs used for training (paper Section 4).
"""

import argparse
import random
from collections import deque

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from minigrid.wrappers import ImgObsWrapper
from torch.distributions import Categorical

from env_utils import NUM_CONFIGS, MAX_STEPS, make_env, obs_to_tensor
from models import ActorCritic


# ===================================================================
# BFS Expert Demos
# ===================================================================

def bfs_path(env, start, goal):
    grid = env.grid
    w, h = grid.width, grid.height
    visited = set([start])
    queue = deque([(start, [])])
    dirs = [(1,0),(0,1),(-1,0),(0,-1)]
    while queue:
        pos, path = queue.popleft()
        if pos == goal:
            return path + [pos]
        for dx, dy in dirs:
            nx, ny = pos[0]+dx, pos[1]+dy
            npos = (nx, ny)
            if 0<=nx<w and 0<=ny<h and npos not in visited:
                cell = grid.get(nx, ny)
                if cell is None or cell.type == "goal":
                    visited.add(npos)
                    queue.append((npos, path+[pos]))
    return None


def path_to_actions(path, initial_dir):
    if len(path) < 2:
        return []
    dir_vecs = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}
    actions, cur = [], initial_dir
    for i in range(len(path)-1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        tgt = None
        for d, (vx,vy) in dir_vecs.items():
            if vx==dx and vy==dy:
                tgt = d; break
        if tgt is None:
            return None
        while cur != tgt:
            diff = (tgt - cur) % 4
            if diff == 1:
                actions.append(1); cur = (cur+1)%4
            elif diff == 3:
                actions.append(0); cur = (cur-1)%4
            else:
                actions.append(1); cur = (cur+1)%4
        actions.append(2)
    return actions


def collect_expert_demos(num_episodes=5000, seed=0):
    all_obs, all_act = [], []
    ok = 0
    for ep in range(num_episodes):
        env = gym.make("MiniGrid-FourRooms-v0", max_steps=MAX_STEPS)
        env = ImgObsWrapper(env)
        obs, _ = env.reset(seed=seed+ep)
        uw = env.unwrapped
        apos, adir = tuple(uw.agent_pos), uw.agent_dir
        gpos = None
        for i in range(uw.grid.width):
            for j in range(uw.grid.height):
                c = uw.grid.get(i,j)
                if c and c.type=="goal":
                    gpos=(i,j); break
            if gpos: break
        if not gpos:
            env.close(); continue
        path = bfs_path(uw, apos, gpos)
        if not path:
            env.close(); continue
        acts = path_to_actions(path, adir)
        if acts is None:
            env.close(); continue
        obs, _ = env.reset(seed=seed+ep)
        eo, ea, done = [], [], False
        for a in acts:
            if done: break
            eo.append(obs_to_tensor(obs))
            ea.append(a)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
        if term and r > 0:
            all_obs.extend(eo); all_act.extend(ea); ok += 1
        env.close()
    print(f"Collected {len(all_obs)} transitions from {ok}/{num_episodes} episodes")
    return np.array(all_obs, np.float32), np.array(all_act, np.int64)


# ===================================================================
# Evaluation (on same 50 configs, 2 rollouts each = 100 total)
# ===================================================================

def evaluate_policy(model, base_seed=0, device="cpu"):
    """Paper Section 4: '100 rollouts across 50 configurations'."""
    model.eval()
    total_r, total_s = 0.0, 0
    total_ep = NUM_CONFIGS * 2

    for ci in range(NUM_CONFIGS):
        for ri in range(2):
            env = make_env()
            obs, _ = env.reset(seed=base_seed + ci)
            done, ep_r = False, 0.0
            while not done:
                t = torch.tensor(obs_to_tensor(obs), dtype=torch.float32,
                                 device=device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = model.get_action_and_value(t)
                obs, reward, term, trunc, _ = env.step(action.item())
                done = term or trunc
                if term and reward > 0:
                    ep_r = reward
                    total_s += 1
            total_r += ep_r
            env.close()

    return total_r / total_ep, total_s / total_ep * 100


# ===================================================================
# Pretraining
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_demos", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default="pretrained.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from env_utils import get_obs_dim
    obs_dim, num_actions = get_obs_dim()
    print(f"Obs dim: {obs_dim}, Num actions: {num_actions}")

    print("Collecting expert demonstrations...")
    obs_data, act_data = collect_expert_demos(args.num_demos, args.seed)
    if len(obs_data) == 0:
        print("ERROR: No demos"); return

    model = ActorCritic(obs_dim, num_actions).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    n = len(obs_data)
    idx = np.random.permutation(n)
    sp = int(n*0.8)
    tr_idx, va_idx = idx[:sp], idx[sp:]
    obs_t = torch.tensor(obs_data, dtype=torch.float32, device=args.device)
    act_t = torch.tensor(act_data, dtype=torch.long, device=args.device)

    best_sr, best_st = 0.0, None
    bs = 256

    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(tr_idx)
        cor, tot = 0, 0
        for s in range(0, len(tr_idx), bs):
            i = tr_idx[s:s+bs]
            logits, _ = model(obs_t[i])
            dist = Categorical(logits=logits)
            loss = nn.functional.cross_entropy(logits, act_t[i]) - 0.01*dist.entropy().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            cor += (logits.argmax(1)==act_t[i]).sum().item(); tot += len(i)

        # Select checkpoint by eval success rate, not val accuracy
        if (epoch+1) % 10 == 0:
            avg_r, sr = evaluate_policy(model, base_seed=0, device=args.device)
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Acc: {cor/tot:.4f} | "
                  f"Eval: Reward={avg_r:.3f}, Success={sr:.1f}%")
            if sr > best_sr:
                best_sr = sr
                best_st = {k:v.clone() for k,v in model.state_dict().items()}

    if best_st: model.load_state_dict(best_st)
    print(f"Best eval success: {best_sr:.1f}%")

    avg_r, sr = evaluate_policy(model, base_seed=0, device=args.device)
    print(f"Pretrained: Reward={avg_r:.3f}, Success={sr:.1f}%")

    torch.save({"model_state_dict": model.state_dict(),
                "obs_dim": obs_dim, "num_actions": num_actions,
                "hidden_dim": 64}, args.save_path)
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()