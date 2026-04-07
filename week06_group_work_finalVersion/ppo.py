"""
PPO for MiniGrid FourRooms.
gamma=0.99, 50 fixed configs, ImgObsWrapper, custom reward 1-0.5*t/H.
Hyperparams: clip=0.2, gae_lambda=0.95, epochs=10, lr=2.5e-4.
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


def train_ppo(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.pretrained_path, map_location=device,
                       weights_only=False)
    obs_dim = ckpt["obs_dim"]
    model = ActorCritic(obs_dim, ckpt["num_actions"],
                        ckpt["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    num_envs = NUM_CONFIGS
    batch_size = num_envs * args.num_steps
    mbs = args.minibatch_size

    envs = [make_env(seed=i) for i in range(num_envs)]
    ol = [obs_to_tensor(e.unwrapped.gen_obs()["image"]) for e in envs]
    next_obs = torch.tensor(np.array(ol), dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)

    print(f"=== PPO ===")
    print(f"Iterations: {args.num_iterations}, Envs: {num_envs}, "
          f"Steps: {args.num_steps}, Epochs: {args.update_epochs}, "
          f"MB: {mbs}")
    best_sr, best_st = 0.0, None

    for it in range(1, args.num_iterations+1):
        frac = 1.0 - (it-1)/args.num_iterations
        for pg in optimizer.param_groups: pg["lr"] = args.lr * frac

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

        model.eval()
        for step in range(args.num_steps):
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            with torch.no_grad():
                a, lp, _, v = model.get_action_and_value(next_obs)
            act_buf[step] = a; logp_buf[step] = lp; val_buf[step] = v
            nl = []
            for ei in range(num_envs):
                o, r, te, tr, _ = envs[ei].step(a[ei].item())
                d = te or tr
                rew_buf[step, ei] = r
                next_done[ei] = float(d)
                if d: o, _ = envs[ei].reset()
                nl.append(obs_to_tensor(o))
            next_obs = torch.tensor(np.array(nl), dtype=torch.float32,
                                    device=device)

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

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1)
        b_lp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_val = val_buf.reshape(-1)

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
                loss = pg + 0.5*vl - args.ent_coef*ent.mean()
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

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
    print(f"\nFinal PPO: Reward={ar:.3f}, Success={sr:.1f}%")
    ckpt["model_state_dict"] = model.state_dict()
    torch.save(ckpt, args.save_path)
    print(f"Saved to {args.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default="pretrained.pt")
    parser.add_argument("--save_path", default="ppo.pt")
    parser.add_argument("--num_iterations", type=int, default=500)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    train_ppo(args)


if __name__ == "__main__":
    main()
