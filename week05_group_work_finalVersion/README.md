# A2C for Continuous Action Spaces — MuJoCo

## Overview

This extends the in-class A2C (CartPole, discrete actions) to continuous action spaces using a **Gaussian policy** and runs on three MuJoCo environments.

Team: Melody Zhao(Yan Zhao), Jason Wang, Zhenyu Dai.

## Key Design Decisions

### 1. Gaussian Policy (Continuous Actions)
- The actor outputs **μ(s)** via a 2-layer MLP (64 hidden, Tanh activations)
- **log σ** is a learnable parameter (state-independent), as in Stable-Baselines3
- Action distribution: `π(a|s) = N(μ(s), σ²I)`
- `log_prob` is summed over action dimensions

### 2. Unified GAE Framework for TD1 / MC / GAE
All three methods use the **same `compute_gae` function** — you only change **λ**:

| Method | λ value | Properties |
|--------|---------|------------|
| TD(1-step) | λ = 0 | Low variance, high bias |
| GAE | λ = 0.95 | Bias-variance trade-off |
| Monte Carlo | λ = 1 | Unbiased, high variance |

**Critical: bootstrap at rollout truncation.** MuJoCo episodes are 1000 steps, but our rollout is only 2048/num_envs ≈ 256 steps per env. When the rollout ends mid-episode, we bootstrap with `V(s_T)` so we don't treat truncation as terminal (return = 0). This is done via `next_values[-1] = last_value`, and the `(1-done)` term handles actual episode terminations correctly. All three methods share this logic — matching the professor's original code and SB3's implementation.

### 3. Network Architecture
- **Actor**: Linear(obs→64) → Tanh → Linear(64→64) → Tanh → Linear(64→act_dim)
- **Critic**: Same architecture but outputs scalar V(s)
- Both use `float64` for numerical stability (matching original code)

## How to Run

```bash
# Install dependencies
pip install gymnasium[mujoco] torch numpy matplotlib

# Run learning curves (3 envs × 3 methods × 3 seeds)
python a2c_continuous.py --mode curves --epochs 300

# Run grid search on a specific environment
python a2c_continuous.py --mode grid --env HalfCheetah-v5

# Run everything
python a2c_continuous.py --mode all

# Fewer epochs for quick test
python a2c_continuous.py --mode curves --epochs 50
```

## Output

Results are saved to `results/`:
- `learning_curves_<env>.png` — one plot per env with TD1/GAE/MC curves (mean ± std across seeds)
- `grid_search_<env>.json` — all grid search configs ranked by score

## Grid Search Space

| Hyperparameter | Values |
|----------------|--------|
| num_envs | 4, 8, 16 |
| policy lr | 1e-4, 3e-4, 1e-3 |
| value lr | 1e-3, 3e-3 |
| γ | 0.99, 0.995 |
| λ | 0.0 (TD1), 0.95 (GAE), 1.0 (MC) |

Total: 3 × 3 × 2 × 2 × 3 = **108 configurations** per environment.

Score = average episodic return over the last 20% of training epochs.

## Environments

| Environment | obs_dim | act_dim | Max Episode Length |
|-------------|---------|---------|-------------------|
| HalfCheetah-v5 | 17 | 6 | 1000 |
| Hopper-v5 | 11 | 3 | 1000 |
| Walker2d-v5 | 17 | 6 | 1000 |
