"""
A2C (Advantage Actor-Critic) for Continuous Action Spaces

Extended from the in-class CartPole (discrete) implementation to handle
MuJoCo continuous-control tasks.

Three advantage estimation methods, all within the GAE framework:
  - TD(1-step):  λ = 0   → Â_t = δ_t  (low variance, high bias)
  - GAE:         λ ∈ (0,1) → Â_t = Σ (γλ)^l δ_{t+l}
  - Monte Carlo: λ = 1   → Â_t = G_t - V(s_t)  (unbiased, high variance)

All three correctly bootstrap at rollout truncation boundaries with V(s_T).

Usage:
  # Run learning curves for 3 MuJoCo envs (TD1 / MC / GAE):
  python a2c_continuous.py --mode curves

  # Run grid search on a specific env:
  python a2c_continuous.py --mode grid --env HalfCheetah-v5

  # Run everything (curves + grid search on all 3 envs):
  python a2c_continuous.py --mode all
"""

import argparse
import copy
import itertools
import json
import os
from typing import SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

NDArrayFloat = npt.NDArray[np.floating]
NDArrayBool = npt.NDArray[np.bool_]

# Networks

class GaussianPolicy:
    """
    Actor for continuous action spaces.
    Outputs μ(s) via a neural network; σ is a learnable parameter (state-independent).
    Action distribution: π(a|s) = N(μ(s), σ²I)
    """

    def __init__(self, env: gym.Env, lr: float = 3e-4, hidden: int = 64) -> None:
        self.num_states: int = env.observation_space.shape[0]
        self.num_actions: int = env.action_space.shape[0]

        # Mean network: s → μ(s)
        self.mu_net: nn.Sequential = nn.Sequential(
            nn.Linear(self.num_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.num_actions),
        ).double()

        # Log standard deviation (state-independent, learnable)
        self.log_std: nn.Parameter = nn.Parameter(
            torch.zeros(self.num_actions, dtype=torch.float64)
        )

        self.opt: Adam = Adam(
            list(self.mu_net.parameters()) + [self.log_std], lr=lr
        )

    def pi(self, s_t: torch.Tensor | NDArrayFloat) -> Normal:
        """Return N(μ(s), σ²I)."""
        s_t_tensor = torch.as_tensor(s_t).double()
        mu = self.mu_net(s_t_tensor)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def act(self, s_t: NDArrayFloat) -> torch.Tensor:
        """Sample a_t ~ π(·|s_t). No grad needed during rollout."""
        with torch.no_grad():
            return self.pi(s_t).sample()

    def learn(
        self, states: NDArrayFloat, actions: NDArrayFloat, advantages: NDArrayFloat
    ) -> torch.Tensor:
        """
        Policy gradient: minimize -E[log π(a|s) · Â(s,a)].
        For multi-dimensional actions, log_prob is summed across action dims.
        """
        actions_t = torch.as_tensor(actions).double()
        advantages_t = torch.as_tensor(advantages).double()

        dist = self.pi(states)
        # Sum log_prob over action dimensions: log π(a|s) = Σ_i log π(a_i|s)
        log_prob = dist.log_prob(actions_t).sum(dim=-1)

        loss = torch.mean(-log_prob * advantages_t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss


class ValueEstimator:
    """Critic network: V(s) approximator, trained by MSE against return targets."""

    def __init__(self, env: gym.Env, lr: float = 1e-3, hidden: int = 64) -> None:
        self.num_states: int = env.observation_space.shape[0]
        self.V: nn.Sequential = nn.Sequential(
            nn.Linear(self.num_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        ).double()
        self.opt: Adam = Adam(self.V.parameters(), lr=lr)

    def predict(self, s_t: NDArrayFloat | torch.Tensor) -> torch.Tensor:
        s_t_tensor = torch.as_tensor(s_t).double()
        return self.V(s_t_tensor).squeeze(-1)

    def learn(self, v_pred: torch.Tensor, returns: NDArrayFloat) -> torch.Tensor:
        returns_t = torch.as_tensor(returns).double()
        loss = torch.mean((v_pred - returns_t) ** 2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss


# Vectorized Environment Wrapper

class VectorizedEnvWrapper(gym.Wrapper):
    """Runs N independent env copies in lockstep for parallel trajectory collection."""

    def __init__(self, env: gym.Env, num_envs: int = 1) -> None:
        super().__init__(env)
        self.num_envs: int = num_envs
        self.envs: list[gym.Env] = [copy.deepcopy(env) for _ in range(num_envs)]

    def reset_all(self) -> NDArrayFloat:
        return np.asarray([env.reset()[0] for env in self.envs])

    def step(
        self, actions: torch.Tensor | NDArrayFloat
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayBool]:
        if isinstance(actions, torch.Tensor):
            actions = actions.numpy()

        next_states, rewards, dones_list = [], [], []
        for env, action in zip(self.envs, actions):
            next_state, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            if done:
                next_states.append(env.reset()[0])
            else:
                next_states.append(next_state)
            rewards.append(reward)
            dones_list.append(done)
        return (
            np.asarray(next_states),
            np.asarray(rewards, dtype=np.float64),
            np.asarray(dones_list, dtype=np.float64),
        )


# Advantage Estimation (TD1 / MC / GAE — unified with bootstrap)

def compute_gae(
    rewards: NDArrayFloat,
    values: NDArrayFloat,
    next_values: NDArrayFloat,
    dones: NDArrayFloat,
    gamma: float,
    lam: float,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """
    Generalized Advantage Estimation (Schulman et al., 2016).

    Computes advantages and return targets via backward recursion:
        δ_t = r_t + γ(1-d_t)V(s_{t+1}) - V(s_t)
        Â_t = δ_t + γλ(1-d_t)Â_{t+1}

    Special cases:
        λ = 0  →  TD(1-step):    Â_t = δ_t
        λ = 1  →  Monte Carlo:   Â_t = G_t - V(s_t)  (with bootstrap at truncation)
        0<λ<1  →  GAE:           bias-variance trade-off

    All three correctly bootstrap at rollout boundaries via next_values[-1] = V(s_T).

    Args:
        rewards:     (T, num_envs)
        values:      (T, num_envs) — V(s_t) for t=0..T-1
        next_values: (T, num_envs) — V(s_{t+1}); last entry is V(s_T) bootstrap
        dones:       (T, num_envs)
        gamma:       discount factor
        lam:         GAE λ  (0=TD1, 1=MC, between=GAE)

    Returns:
        advantages: (T, num_envs)
        returns:    (T, num_envs)  — advantage + baseline = G_t targets for critic
    """
    T = rewards.shape[0]
    advantages = np.empty_like(rewards)

    # TD errors: δ_t = r_t + γ(1-d_t)V(s_{t+1}) - V(s_t)
    td_errors = rewards + gamma * (1 - dones) * next_values - values

    # Backward recursion for GAE
    advantages[-1] = td_errors[-1]
    for t in range(T - 2, -1, -1):
        advantages[t] = td_errors[t] + gamma * lam * (1 - dones[t]) * advantages[t + 1]

    returns = advantages + values  # G_t = Â_t + V(s_t)
    return advantages, returns


# A2C Training Loop


def a2c(
    env: VectorizedEnvWrapper,
    agent: GaussianPolicy,
    critic: ValueEstimator,
    gamma: float = 0.99,
    lam: float = 0.95,
    epochs: int = 300,
    train_v_iters: int = 10,
    rollout_len: int = 2048,
) -> list[float]:
    """
    A2C training loop.

    Returns:
        avg_returns: list of average episodic returns per epoch
    """
    num_states = agent.num_states
    num_envs = env.num_envs

    # Pre-allocate buffers: [time, num_envs, ...]
    states = np.empty((rollout_len, num_envs, num_states))
    actions = np.empty((rollout_len, num_envs, agent.num_actions))
    rewards = np.empty((rollout_len, num_envs))
    dones = np.empty((rollout_len, num_envs))

    avg_returns: list[float] = []
    s_t = env.reset_all()

    for epoch in range(epochs):
        #  Phase 1: Collect rollout 
        epoch_rewards = []
        epoch_ep_returns = []
        running_ep_reward = np.zeros(num_envs)

        for t in range(rollout_len):
            a_t = agent.act(s_t)
            s_next, r_t, d_t = env.step(a_t)
            states[t] = s_t
            actions[t] = a_t.numpy() if isinstance(a_t, torch.Tensor) else a_t
            rewards[t] = r_t
            dones[t] = d_t

            running_ep_reward += r_t
            for i in range(num_envs):
                if d_t[i]:
                    epoch_ep_returns.append(running_ep_reward[i])
                    running_ep_reward[i] = 0.0

            s_t = s_next

        # Phase 2: Compute value predictions 
        with torch.no_grad():
            # V(s_t) for all rollout steps
            flat_states = states.reshape(-1, num_states)
            values = critic.predict(flat_states).numpy().reshape(rollout_len, num_envs)

            # V(s_{t+1}): shift values by 1, bootstrap last with V(s_T)
            last_value = critic.predict(s_t).numpy()  # V(s_T) for bootstrap
            next_values = np.empty_like(values)
            next_values[:-1] = values[1:]
            # At rollout boundary: if episode ended, V=0; else bootstrap with V(s_T)
            # This is handled by the (1-done) term, but we set next_values[-1] = last_value
            # so that non-terminal truncations get proper bootstrap
            next_values[-1] = last_value

        # Phase 3: Compute advantages & returns 
        advantages, returns = compute_gae(
            rewards, values, next_values, dones, gamma, lam
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Phase 4: Update critic 
        flat_returns = returns.reshape(-1)
        for _ in range(train_v_iters):
            v_pred = critic.predict(flat_states)
            critic.learn(v_pred, flat_returns)

        # Phase 5: Update actor 
        flat_actions = actions.reshape(-1, agent.num_actions)
        flat_advantages = advantages.reshape(-1)
        agent.learn(flat_states, flat_actions, flat_advantages)

        # Logging 
        if len(epoch_ep_returns) > 0:
            avg_ret = float(np.mean(epoch_ep_returns))
        else:
            avg_ret = float(rewards.sum() / max(dones.sum(), 1))
        avg_returns.append(avg_ret)
        print(f"  Epoch {epoch+1}/{epochs}  |  Avg Return: {avg_ret:.1f}")

    env.close()
    return avg_returns



# Learning Curve Experiment

def run_learning_curves(
    env_names: list[str],
    seeds: list[int] = [0, 1, 2],
    epochs: int = 300,
    rollout_len: int = 2048,
    num_envs: int = 8,
    pi_lr: float = 3e-4,
    v_lr: float = 1e-3,
    gamma: float = 0.99,
    output_dir: str = "results",
) -> None:
    """
    For each env, run TD1 (λ=0), GAE (λ=0.95), MC (λ=1) across multiple seeds.
    Produces one plot per env with all three methods.
    """
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        "TD1 (λ=0)": 0.0,
        "GAE (λ=0.95)": 0.95,
        "MC  (λ=1)": 1.0,
    }

    for env_name in env_names:
        print(f"\n{'='*60}")
        print(f"  Environment: {env_name}")
        print(f"{'='*60}")

        all_curves: dict[str, list[list[float]]] = {m: [] for m in methods}

        for method_name, lam in methods.items():
            for seed in seeds:
                print(f"\n--- {method_name} | seed={seed} ---")
                torch.manual_seed(seed)
                np.random.seed(seed)

                base_env = gym.make(env_name)
                base_env.reset(seed=seed)
                env = VectorizedEnvWrapper(base_env, num_envs=num_envs)

                agent = GaussianPolicy(env, lr=pi_lr)
                critic = ValueEstimator(env, lr=v_lr)
                returns = a2c(
                    env, agent, critic,
                    gamma=gamma, lam=lam,
                    epochs=epochs, rollout_len=rollout_len,
                    train_v_iters=10,
                )
                all_curves[method_name].append(returns)

        # Plot 
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"TD1 (λ=0)": "#e74c3c", "GAE (λ=0.95)": "#2ecc71", "MC  (λ=1)": "#3498db"}

        for method_name, curves in all_curves.items():
            arr = np.array(curves)  # (num_seeds, epochs)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, label=method_name, color=colors[method_name], linewidth=2)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=colors[method_name])

        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("Average Episodic Return", fontsize=13)
        ax.set_title(f"A2C Learning Curves — {env_name}", fontsize=15)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(output_dir, f"learning_curves_{env_name.replace('-', '_')}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"\nPlot saved → {plot_path}")


# Grid Search

def _cfg_key(cfg: dict) -> str:
    """Deterministic string key for a config dict, used to check if already completed."""
    return json.dumps({k: cfg[k] for k in sorted(cfg) if k != "score"}, sort_keys=True)


def run_grid_search(
    env_name: str,
    epochs: int = 150,
    rollout_len: int = 2048,
    output_dir: str = "results",
) -> None:
    """
    Hyperparameter grid search over: num_envs, pi_lr, v_lr, γ, λ.
    Evaluates each config by average return over the last 20% of training.

    ** Checkpoint / resume: **
    Results are saved incrementally to a JSON-Lines (.jsonl) file after each
    config finishes. If the script is restarted, already-completed configs are
    skipped automatically. A final sorted summary JSON is also written at the end.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Search grid (keep small for tractability) 
    grid = {
        "num_envs":  [4, 8, 16],
        "pi_lr":     [1e-4, 3e-4, 1e-3],
        "v_lr":      [1e-3, 3e-3],
        "gamma":     [0.99, 0.995],
        "lam":       [0.0, 0.95, 1.0],  # TD1 / GAE / MC
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    total = len(combos)

    # Load checkpoint (if any) 
    checkpoint_path = os.path.join(
        output_dir, f"grid_search_{env_name.replace('-', '_')}_checkpoint.jsonl"
    )
    completed: dict[str, dict] = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    completed[_cfg_key(entry)] = entry
        print(f"Resuming grid search: {len(completed)}/{total} already completed")
    else:
        print(f"\nGrid search on {env_name}: {total} configurations")

    results: list[dict] = list(completed.values())

    for i, combo in enumerate(combos):
        cfg = dict(zip(keys, combo))
        key = _cfg_key(cfg)

        # Skip if already done
        if key in completed:
            print(f"[{i+1}/{total}] SKIP (already done) {cfg}")
            continue

        print(f"\n[{i+1}/{total}] {cfg}")

        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        base_env = gym.make(env_name)
        base_env.reset(seed=seed)
        env = VectorizedEnvWrapper(base_env, num_envs=cfg["num_envs"])

        agent = GaussianPolicy(env, lr=cfg["pi_lr"])
        critic = ValueEstimator(env, lr=cfg["v_lr"])
        returns = a2c(
            env, agent, critic,
            gamma=cfg["gamma"], lam=cfg["lam"],
            epochs=epochs, rollout_len=rollout_len,
            train_v_iters=10,
        )

        # Score = mean return over last 20% of epochs
        tail = max(1, len(returns) // 5)
        score = float(np.mean(returns[-tail:]))
        entry = {**cfg, "score": score, "all_returns": returns}
        results.append(entry)
        print(f"  → Score (last 20% avg): {score:.1f}")

        # Incremental save (append one line) 
        with open(checkpoint_path, "a") as f:
            # Save a compact version (without full returns curve) for checkpoint
            f.write(json.dumps({**cfg, "score": score}) + "\n")

    # Final sorted summary 
    # Strip 'all_returns' for the summary JSON (too large)
    summary = [{k: v for k, v in r.items() if k != "all_returns"} for r in results]
    summary.sort(key=lambda r: r["score"], reverse=True)

    report_path = os.path.join(output_dir, f"grid_search_{env_name.replace('-', '_')}.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Grid Search Results — {env_name}  (Top 5)")
    print(f"{'='*60}")
    for rank, r in enumerate(summary[:5], 1):
        print(f"  #{rank}  Score={r['score']:.1f}  |  num_envs={r['num_envs']}  "
              f"pi_lr={r['pi_lr']}  v_lr={r['v_lr']}  γ={r['gamma']}  λ={r['lam']}")

    print(f"\nFull results saved → {report_path}")
    print(f"Checkpoint file   → {checkpoint_path}")

# Main

ENV_NAMES = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]


def main():
    parser = argparse.ArgumentParser(description="A2C for Continuous Action Spaces")
    parser.add_argument(
        "--mode", choices=["curves", "grid", "all"], default="curves",
        help="'curves': learning curve comparison; 'grid': hyperparameter search; 'all': both"
    )
    parser.add_argument(
        "--env", type=str, default=None,
        help="Specific env for grid search (default: all three)"
    )
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    if args.mode in ("curves", "all"):
        run_learning_curves(
            env_names=ENV_NAMES,
            seeds=args.seeds,
            epochs=args.epochs,
            output_dir=args.output,
        )

    if args.mode in ("grid", "all"):
        envs = [args.env] if args.env else ENV_NAMES
        for env_name in envs:
            run_grid_search(
                env_name=env_name,
                epochs=min(args.epochs, 150),
                output_dir=args.output,
            )


if __name__ == "__main__":
    main()
