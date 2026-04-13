"""
REINFORCE with Baseline for MiniGrid FourRooms (RLFT setting).
Same batch rollout structure as PPO but:
  - Single epoch update (no reuse of data)
  - No clipping (vanilla policy gradient)
This matches the paper's RLFT setup for REINFORCE with baseline.
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env_utils import NUM_CONFIGS, make_env, obs_to_tensor
from models import ActorCritic
from pretrain import evaluate_policy


def train_reinforce(args):
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    num_envs = NUM_CONFIGS
    batch_size = num_envs * args.num_steps

    envs = [make_env(seed=i) for i in range(num_envs)]
    ol = [obs_to_tensor(e.unwrapped.gen_obs()["image"]) for e in envs]
    next_obs = torch.tensor(np.array(ol), dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)

    print(f"=== REINFORCE with Baseline (RLFT) ===")
    print(f"Iterations: {args.num_iterations}, Envs: {num_envs}, "
          f"Steps: {args.num_steps}, lr: {args.lr}")
    best_sr, best_st = 0.0, None

    for it in range(1, args.num_iterations + 1):
        frac = 1.0 - (it - 1) / args.num_iterations
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * frac

        obs_buf = torch.zeros((args.num_steps, num_envs, obs_dim),
                              dtype=torch.float32, device=device)
        act_buf = torch.zeros((args.num_steps, num_envs),
                              dtype=torch.long, device=device)
        rew_buf = torch.zeros((args.num_steps, num_envs),
                              dtype=torch.float32, device=device)
        done_buf = torch.zeros((args.num_steps, num_envs),
                               dtype=torch.float32, device=device)
        val_buf = torch.zeros((args.num_steps, num_envs),
                              dtype=torch.float32, device=device)

        model.eval()
        for step in range(args.num_steps):
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            with torch.no_grad():
                action, _, _, value = model.get_action_and_value(next_obs)
            act_buf[step] = action
            val_buf[step] = value

            nl = []
            for ei in range(num_envs):
                o, r, te, tr, _ = envs[ei].step(action[ei].item())
                d = te or tr
                rew_buf[step, ei] = r
                next_done[ei] = float(d)
                if d:
                    o, _ = envs[ei].reset()
                nl.append(obs_to_tensor(o))
            next_obs = torch.tensor(np.array(nl), dtype=torch.float32,
                                    device=device)

        # GAE
        with torch.no_grad():
            nv = model.get_value(next_obs)
        adv = torch.zeros_like(rew_buf)
        lg = torch.zeros(num_envs, dtype=torch.float32, device=device)
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nnt, nvt = 1.0 - next_done, nv
            else:
                nnt, nvt = 1.0 - done_buf[t + 1], val_buf[t + 1]
            delta = rew_buf[t] + args.gamma * nvt * nnt - val_buf[t]
            lg = delta + args.gamma * args.gae_lambda * nnt * lg
            adv[t] = lg
        ret = adv + val_buf

        # Flatten
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)

        # Normalize advantages
        if b_adv.std() > 1e-8:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # Single epoch, no clipping = REINFORCE with baseline
        model.train()
        _, nlp, ent, nval = model.get_action_and_value(b_obs, b_act)

        # Vanilla policy gradient loss (no clipping)
        policy_loss = -(nlp * b_adv.detach()).mean()
        value_loss = 0.5 * ((nval - b_ret.detach()) ** 2).mean()

        loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * ent.mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if it % args.eval_interval == 0:
            ar, sr = evaluate_policy(model, base_seed=0, device=args.device)
            print(f"Iter {it}/{args.num_iterations} | "
                  f"Reward: {ar:.3f} | Success: {sr:.1f}%")
            if sr > best_sr:
                best_sr = sr
                best_st = {k: v.clone() for k, v in model.state_dict().items()}

    for e in envs:
        e.close()
    if best_st:
        model.load_state_dict(best_st)
    ar, sr = evaluate_policy(model, base_seed=0, device=args.device)
    print(f"\nFinal REINFORCE: Reward={ar:.3f}, Success={sr:.1f}%")

    ckpt["model_state_dict"] = model.state_dict()
    torch.save(ckpt, args.save_path)
    print(f"Saved to {args.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default="pretrained.pt")
    parser.add_argument("--save_path", default="reinforce.pt")
    parser.add_argument("--num_iterations", type=int, default=500)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=1)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train_reinforce(args)


if __name__ == "__main__":
    main()