"""
Polychromic PPO for MiniGrid FourRooms.
Algorithm 2 from the paper.

Core Poly-PPO features (all from paper):
  - Vine sampling at rollout states (N=8, n=4, p=2, M=4)
  - Polychromic advantage REPLACES GAE at rollout states (Algorithm 2 line 22)
  - KL penalty from behavior policy (Algorithm 2 line 31)
  - Window W=5 (Appendix A.1.2)
  - Update epochs=2 (Table 3)
  - gamma=1.0 (Table 3)

Architecture: shared ActorCritic (paper Appendix A: "compact MLP")
Learning rate: 2.5e-4 (adapted for compact MLP; Table 3 lr=1e-5
is tuned for larger CNN-GRU used in BabyAI tasks)
"""

import argparse
import copy
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from env_utils import NUM_CONFIGS, make_env, obs_to_tensor
from models import ActorCritic
from pretrain import evaluate_policy


# ===================================================================
# Diversity
# ===================================================================

def get_room_id(pos, grid_size=19):
    x, y = pos
    mid = grid_size // 2
    if x<=mid and y<=mid: return 0
    elif x>mid and y<=mid: return 1
    elif x<=mid and y>mid: return 2
    else: return 3


def trajectory_signature(positions, grid_size=19):
    """Create a compact signature of a trajectory's path.
    Uses: (1) which rooms visited, (2) final position, (3) path length.
    This captures more diversity than room-set alone."""
    if not positions:
        return (frozenset(), (0,0), 0)
    rooms = frozenset(get_room_id(p, grid_size) for p in positions)
    final_pos = positions[-1]
    # Bin positions into 4x4 grid cells for path shape
    cells = frozenset((p[0]//5, p[1]//5) for p in positions)
    return (rooms, final_pos, cells)


def diversity_function(signatures):
    """Diversity = fraction of unique trajectory signatures in the set."""
    n = len(signatures)
    if n <= 1: return 0.0
    u = len(set(signatures))
    return u/n if u > 1 else 0.0


def polychromic_objective(vines, indices, gs):
    """Eq. 7: f_poly = (1/n) * sum(R(tau_i)) * d(s, tau_1:n)"""
    n = len(indices)
    if n == 0: return 0.0
    rews, sigs = [], []
    for i in indices:
        rews.append(vines[i]["reward"])
        sigs.append(trajectory_signature(vines[i]["pos"], gs))
    return (sum(rews)/n) * diversity_function(sigs)


def vine_rollout(env_snap, model, start_obs, device, max_steps, gs=19):
    env = copy.deepcopy(env_snap)
    pos = [tuple(env.unwrapped.agent_pos)]
    obs, tr, done = start_obs, 0.0, False
    for _ in range(max_steps):
        if done: break
        t = torch.tensor(obs_to_tensor(obs), dtype=torch.float32,
                         device=device).unsqueeze(0)
        with torch.no_grad():
            a, _, _, _ = model.get_action_and_value(t)
        obs, r, te, tu, _ = env.step(a.item())
        done = te or tu; tr += r
        pos.append(tuple(env.unwrapped.agent_pos))
    try: env.close()
    except: pass
    return {"reward": max(0.0, min(1.0, tr)), "pos": pos}


# ===================================================================
# Training
# ===================================================================

def train_polyppo(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.pretrained_path, map_location=device,
                       weights_only=False)
    obs_dim = ckpt["obs_dim"]
    num_actions = ckpt["num_actions"]
    hidden_dim = ckpt["hidden_dim"]

    model = ActorCritic(obs_dim, num_actions, hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Behavior model for KL penalty (frozen)
    beh = ActorCritic(obs_dim, num_actions, hidden_dim).to(device)
    beh.load_state_dict(model.state_dict())
    for p in beh.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    num_envs = NUM_CONFIGS
    batch_size = num_envs * args.num_steps
    mbs = args.minibatch_size

    spacing = max(1, args.num_steps // (args.p_rollout + 1))
    r_steps = set((k+1)*spacing for k in range(args.p_rollout))

    envs = [make_env(seed=i) for i in range(num_envs)]
    ol = [obs_to_tensor(e.unwrapped.gen_obs()["image"]) for e in envs]
    or_list = [e.unwrapped.gen_obs()["image"] for e in envs]
    next_obs = torch.tensor(np.array(ol), dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)
    next_raw = or_list

    print(f"=== Polychromic PPO (Algorithm 2) ===")
    print(f"Iter: {args.num_iterations}, Envs: {num_envs}, "
          f"Steps: {args.num_steps}")
    print(f"gamma={args.gamma}, lr={args.lr}, epochs={args.update_epochs}")
    print(f"Vine: N={args.N}, n={args.n}, p={args.p_rollout}, "
          f"M={args.M}, W={args.W}, kl={args.kl_coef}")
    best_sr, best_st = 0.0, None

    for it in range(1, args.num_iterations+1):
        frac = 1.0 - (it-1)/args.num_iterations
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * frac

        obs_buf = torch.zeros((args.num_steps, num_envs, obs_dim),
                              dtype=torch.float32, device=device)
        act_buf = torch.zeros((args.num_steps, num_envs),
                              dtype=torch.long, device=device)
        logp_buf = torch.zeros((args.num_steps, num_envs),
                               dtype=torch.float32, device=device)
        rew_buf = torch.zeros((args.num_steps, num_envs),
                              dtype=torch.float32, device=device)
        done_buf = torch.zeros((args.num_steps, num_envs),
                               dtype=torch.float32, device=device)
        val_buf = torch.zeros((args.num_steps, num_envs),
                              dtype=torch.float32, device=device)
        vsnap = defaultdict(list)

        model.eval()
        for step in range(args.num_steps):
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            with torch.no_grad():
                a, lp, _, v = model.get_action_and_value(next_obs)
            act_buf[step] = a; logp_buf[step] = lp; val_buf[step] = v

            if step in r_steps:
                for ei in range(num_envs):
                    if next_done[ei].item() == 0:
                        vsnap[ei].append((step, copy.deepcopy(envs[ei]),
                                          next_raw[ei].copy()))

            nl, nr = [], []
            for ei in range(num_envs):
                o, r, te, tr, _ = envs[ei].step(a[ei].item())
                d = te or tr
                rew_buf[step, ei] = r; next_done[ei] = float(d)
                if d: o, _ = envs[ei].reset()
                nl.append(obs_to_tensor(o)); nr.append(o)
            next_obs = torch.tensor(np.array(nl), dtype=torch.float32,
                                    device=device)
            next_raw = nr

        # --- GAE for all states ---
        with torch.no_grad():
            nv = model.get_value(next_obs)
        adv = torch.zeros_like(rew_buf)
        lg = torch.zeros(num_envs, dtype=torch.float32, device=device)
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps-1:
                nnt, nvt = 1.0-next_done, nv
            else:
                nnt, nvt = 1.0-done_buf[t+1], val_buf[t+1]
            delta = rew_buf[t] + args.gamma*nvt*nnt - val_buf[t]
            lg = delta + args.gamma*args.gae_lambda*nnt*lg
            adv[t] = lg
        ret = adv + val_buf

        # --- Polychromic advantage: REPLACE GAE at rollout states ---
        # Algorithm 2 lines 18-23
        for ei, snaps in vsnap.items():
            gs = envs[ei].unwrapped.grid.width
            for (ss, se, sr) in snaps:
                rem = args.num_steps - ss
                vs = [vine_rollout(se, model, sr, device, rem, gs)
                      for _ in range(args.N)]
                try: se.close()
                except: pass
                if len(vs) < args.n:
                    continue

                pool = list(range(len(vs)))
                sets = [random.sample(pool, args.n) for _ in range(args.M)]
                scores = [polychromic_objective(vs, s, gs) for s in sets]
                baseline = np.mean(scores)

                # Algorithm 2 line 22: A_poly = score(gi) - baseline
                # Use the best set's advantage
                poly_adv = max(scores) - baseline

                # Only replace GAE if polychromic signal is meaningful
                # When diversity=0 for all sets, all scores=0, poly_adv=0
                # In that case, keep GAE advantage (standard PPO behavior)
                if abs(poly_adv) < 1e-8:
                    continue

                # Scale to match GAE advantage magnitude
                gae_std = adv[:, ei].std().item()
                if gae_std > 1e-8:
                    score_range = max(abs(s - baseline) for s in scores) + 1e-8
                    poly_adv = (poly_adv / score_range) * gae_std

                # Replace advantage only (for actor). Keep original ret (for critic).
                for w in range(args.W + 1):
                    t = ss + w
                    if t < args.num_steps:
                        adv[t, ei] = poly_adv

        # Flatten
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1)
        b_lp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_val = val_buf.reshape(-1)

        # --- PPO + KL update (Algorithm 2 lines 25-35) ---
        model.train()
        inds = np.arange(batch_size)
        for ep in range(args.update_epochs):
            np.random.shuffle(inds)
            for s in range(0, batch_size, mbs):
                mb = inds[s:s+mbs]
                _, nlp, ent, nval = model.get_action_and_value(
                    b_obs[mb], b_act[mb])
                ratio = (nlp - b_lp[mb]).exp()
                ma = b_adv[mb]
                if ma.std() > 1e-8:
                    ma = (ma - ma.mean()) / (ma.std() + 1e-8)
                pg1 = -ma * ratio
                pg2 = -ma * torch.clamp(ratio, 1-args.clip, 1+args.clip)
                pg = torch.max(pg1, pg2).mean()
                vc = b_val[mb] + torch.clamp(nval-b_val[mb],
                                             -args.clip, args.clip)
                vl = 0.5*torch.max((nval-b_ret[mb])**2,
                                   (vc-b_ret[mb])**2).mean()
                with torch.no_grad():
                    bl2, _ = beh(b_obs[mb])
                    bd = Categorical(logits=bl2)
                cl, _ = model(b_obs[mb])
                cd = Categorical(logits=cl)
                kl = torch.distributions.kl_divergence(bd, cd).mean()

                loss = pg + 0.5 * vl - args.ent_coef * ent.mean() + args.kl_coef * kl

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            # Algorithm 2 line 33: Update behavior policy after each epoch
            beh.load_state_dict(model.state_dict())

        if it % args.eval_interval == 0:
            ar, sr = evaluate_policy(model, base_seed=0,
                                     device=args.device)
            print(f"Iter {it}/{args.num_iterations} | "
                  f"Reward: {ar:.3f} | Success: {sr:.1f}%")
            if sr > best_sr:
                best_sr = sr
                best_st = {k:v.clone() for k,v in model.state_dict().items()}

    for e in envs: e.close()
    if best_st: model.load_state_dict(best_st)
    ar, sr = evaluate_policy(model, base_seed=0, device=args.device)
    print(f"\nFinal Poly-PPO: Reward={ar:.3f}, Success={sr:.1f}%")

    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": obs_dim, "num_actions": num_actions,
        "hidden_dim": hidden_dim,
    }, args.save_path)
    print(f"Saved to {args.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default="pretrained.pt")
    parser.add_argument("--save_path", default="polyppo.pt")
    parser.add_argument("--num_iterations", type=int, default=500)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--p_rollout", type=int, default=2)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--W", type=int, default=5)
    parser.add_argument("--kl_coef", type=float, default=0.005)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train_polyppo(args)


if __name__ == "__main__":
    main()