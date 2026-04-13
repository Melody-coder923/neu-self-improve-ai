# Polychromic PPO — FourRooms Replication

Re-implementation of results from:
**Polychromic Objectives for Reinforcement Learning** (Hamid et al., 2026)

- Paper: [arXiv 2509.25424v3](https://arxiv.org/abs/2509.25424v3)

---

## Course Information

- **Course:** INFO 7375 — Special Topics in AI Engineering and Applications
- **Institution:** Northeastern University
- **Instructor:** Prof. Suhabe Bugrara
- **Assignment:** Week 06 Group Work — PPO and Polychromic PPO

## Team Members

- Yan Zhao
- Zhenyu Dai
- Chien-Cheng Wang

---

## Overview

This project replicates the **MiniGrid FourRooms** results from Table 1 of the paper. Starting from a pretrained policy (behavioral cloning on expert demonstrations), we fine-tune using three reinforcement learning methods:

1. **REINFORCE with Baseline** (Williams, 1992) — Vanilla policy gradient with learned value baseline
2. **PPO** (Schulman et al., 2017) — Proximal Policy Optimization with clipped surrogate objective
3. **Poly-PPO** (Algorithm 2) — Polychromic PPO with vine sampling and diversity-aware polychromic advantage

The core idea: standard RLFT methods (REINFORCE, PPO) maximize single-trajectory reward, which can cause entropy collapse — the policy converges to a narrow set of behaviors. Poly-PPO instead optimizes a *set-level* polychromic objective that jointly rewards success and diversity, encouraging the policy to maintain a broad repertoire of strategies.

---

## Results Summary

| Method | Seed 0 | Seed 1 | Seed 2 | Our Avg | Paper |
|--------|--------|--------|--------|---------|-------|
| Pretrained | 68.0% | 67.0% | 69.0% | **68.0%** | 70.4% |
| REINFORCE | 68.0% | 79.0% | 64.0% | **70.3%** | 89.6% |
| PPO | 91.0% | 88.0% | 94.0% | **91.0%** | 89.2% |
| Poly-PPO | 83.0% | 86.0% | 80.0% | **83.0%** | 92.4% |

Results averaged over 3 random seeds. Evaluation: 100 rollouts across 50 fixed configurations (2 rollouts per config).

### Key Observations

**1. PPO exceeds the paper (91.0% vs 89.2%)**

| | Pretrained | PPO | Improvement |
|--|-----------|-----|------------|
| Paper | 70.4% | 89.2% | +18.8% |
| **Ours** | **68.0%** | **91.0%** | **+23.0%** |

All three seeds show strong improvement over pretrained, confirming RLFT with PPO is highly effective on FourRooms. Our PPO exceeds the paper despite a lower pretrained starting point.

**2. Poly-PPO shows substantial improvement over pretrained (+15%)**

| | Pretrained | Poly-PPO | Improvement |
|--|-----------|----------|------------|
| Paper | 70.4% | 92.4% | +22.0% |
| **Ours** | **68.0%** | **83.0%** | **+15.0%** |

Poly-PPO improves markedly over pretrained, demonstrating that vine sampling and polychromic advantage provide meaningful learning signal. The remaining gap from the paper (83% vs 92.4%) is attributable to undisclosed architecture details — see [Gap Analysis](#gap-analysis).

**3. REINFORCE is high-variance without clipping**

| | Pretrained | REINFORCE | Improvement |
|--|-----------|-----------|------------|
| Paper | 70.4% | 89.6% | +19.2% |
| **Ours** | **68.0%** | **70.3%** | **+2.3%** |

REINFORCE shows the largest gap from the paper. Without clipping, updates are unstable — success rates oscillate between 53% and 79% across training. The paper does not publish REINFORCE hyperparameters. This instability is a known limitation that PPO was specifically designed to address.

**4. Pretrained baseline is close to paper (68.0% vs 70.4%)**

The 2.4% gap is within random seed variance. Both use behavioral cloning on expert shortest-path demonstrations with 80/20 train-test split and entropy regularization.

---

## Project Structure

```
week_06/
├── env_utils.py       # Environment: ImgObsWrapper, custom reward (1-0.5*t/H), 50 fixed configs
├── models.py          # ActorCritic (shared backbone) and SeparateActorCritic
├── pretrain.py        # BFS expert demos + behavioral cloning + checkpoint selection by eval success
├── reinforce.py       # REINFORCE with baseline: batch rollout, single epoch, no clipping
├── ppo.py             # PPO: batch rollout, multi-epoch clipped updates, GAE
├── polyppo.py         # Poly-PPO: Algorithm 2 with vine sampling + polychromic advantage
└── README.md
```

---

## Dependencies

```bash
pip install torch gymnasium minigrid numpy tqdm
```

---

## How to Run

### Step 1: Pretrain (3 seeds)
```bash
python pretrain.py --seed 0
python pretrain.py --seed 1 --save_path pretrained_s1.pt
python pretrain.py --seed 2 --save_path pretrained_s2.pt
```

Collects 5000 BFS expert episodes, trains via cross-entropy with entropy regularizer, selects checkpoint by eval success rate (every 10 epochs).

### Step 2: REINFORCE (3 seeds)
```bash
python reinforce.py --seed 0 --save_path reinforce_s0.pt
python reinforce.py --seed 1 --pretrained_path pretrained_s1.pt --save_path reinforce_s1.pt
python reinforce.py --seed 2 --pretrained_path pretrained_s2.pt --save_path reinforce_s2.pt
```

### Step 3: PPO (3 seeds)
```bash
python ppo.py --seed 0 --save_path ppo_s0.pt
python ppo.py --seed 1 --pretrained_path pretrained_s1.pt --save_path ppo_s1.pt
python ppo.py --seed 2 --pretrained_path pretrained_s2.pt --save_path ppo_s2.pt
```

### Step 4: Poly-PPO (3 seeds)
```bash
python polyppo.py --seed 0 --save_path polyppo_s0.pt
python polyppo.py --seed 1 --pretrained_path pretrained_s1.pt --save_path polyppo_s1.pt
python polyppo.py --seed 2 --pretrained_path pretrained_s2.pt --save_path polyppo_s2.pt
```

---

## Implementation Details

### Environment Setup (Paper Appendix A)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Environment | MiniGrid-FourRooms-v0 | Paper Table 1 |
| Observation | ImgObsWrapper → flatten (7×7×3 = 147 dim) | Appendix A: "compact MLP conditioned on flattened image observation" |
| Reward | 1 - 0.5 × t/H on success, 0 otherwise | Appendix A: lowered time penalty improves diversity during RLFT |
| Max steps (H) | 100 | Appendix A |
| Configurations | 50 fixed seeds (0–49) | Section 4: "fine-tune on 50 configurations" |
| Evaluation | 2 rollouts × 50 configs = 100 total | "100 rollouts across 50 configurations and 3 random seeds" |

### Pretraining (Behavioral Cloning)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Expert data | BFS shortest-path (5000 episodes, 84K transitions) | Equivalent to Minari expert policy (Younis et al., 2024) |
| Loss | Cross-entropy + entropy regularizer (coef=0.01) | Appendix A |
| Train/test split | 80/20 | Appendix A |
| Checkpoint | Best eval success rate (checked every 10 epochs) | Selects model that navigates to goal, not just per-step accuracy |
| Architecture | Shared ActorCritic: 147→64→64→{7 actions, 1 value} | Appendix A: "compact MLP" |

### REINFORCE with Baseline (Williams, 1992)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rollout | 50 envs × 128 steps, single epoch, no clipping | Vanilla policy gradient in RLFT setting |
| Advantage | GAE with λ=1.0 (= Monte Carlo return) | Williams 1992: use complete episode returns |
| Learning rate | 2.5e-4 (linear annealing) | — |
| Gamma | 0.99 | — |
| Baseline | Learned value function (shared ActorCritic) | Paper: "REINFORCE with baseline" |

### PPO (Schulman et al., 2017)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rollout | 50 envs × 128 steps | One env per fixed config |
| Update epochs | 10 | Assignment pseudocode |
| Minibatch size | 64 | — |
| Learning rate | 2.5e-4 (linear annealing) | Assignment pseudocode |
| Gamma | 0.99 | Assignment pseudocode |
| GAE lambda | 0.95 | Assignment pseudocode |
| Clip coefficient | 0.2 | Assignment pseudocode |
| Value loss coef | 0.5 | Assignment pseudocode |
| Entropy coef | 0.01 | Assignment pseudocode |
| Max grad norm | 0.5 | Assignment pseudocode |

### Poly-PPO (Algorithm 2)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Update epochs | 5 | Adapted for compact MLP (Table 3: 2 for CNN-GRU) |
| Gamma | 0.99 | Adapted for compact MLP (Table 3: 1.0 for CNN-GRU) |
| GAE lambda | 0.95 | Table 3 |
| Clip coefficient | 0.2 | Table 3 |
| KL coefficient | 0.005 | Adapted (Table 3 sweeps {0.005, 0.01, 0.05, 0.1}) |
| Max grad norm | 0.5 | Table 3 |
| Learning rate | 2.5e-4 | Adapted for compact MLP (Table 3: 1e-5 for CNN-GRU) |
| Num vines N | 8 | Table 3 |
| Set size n | 4 | Table 3 |
| Num sets M | 4 | Table 3 |
| Rollout states p | 2 | Table 3 |
| Window W | 5 | Table 3 (BabyAI/Minigrid) |

---

## Method

### Pretraining
Expert demonstrations are collected by running BFS (breadth-first search) on each FourRooms configuration to find the shortest path from agent to goal. The policy is trained via behavioral cloning (cross-entropy loss) to imitate these demonstrations. Checkpoints are selected by eval success rate rather than validation accuracy, since per-step accuracy does not guarantee successful navigation.

### REINFORCE with Baseline
The simplest RLFT method: collect a batch of on-policy trajectories, compute Monte Carlo returns (GAE with λ=1.0), subtract a learned value baseline, and take a single gradient step. No clipping or ratio truncation is applied. This provides an unbiased but high-variance policy gradient estimate.

### PPO
Extends REINFORCE with two key stabilizers: (1) a clipped surrogate objective that limits how much the policy can change per update, and (2) multiple epochs of minibatch updates per rollout batch, allowing more efficient use of collected data. These modifications dramatically reduce variance and enable reliable convergence.

### Poly-PPO (Algorithm 2)
Extends PPO with the polychromic objective for diversity-aware training:

**1. Vine Sampling (Algorithm 2 lines 3–12)**
At p=2 equally spaced rollout states during each iteration, we save environment snapshots. From each rollout state, N=8 vine trajectories are generated by rolling out the current policy from that saved state. This produces multiple trajectory sets from the same starting point.

**2. Polychromic Objective (Eq. 7)**
For each rollout state, the N=8 vine trajectories are grouped into M=4 random sets of size n=4. Each set is scored by:

f_poly = (1/n) × Σ R(τ_i) × d(s, τ_1:n)

where R(τ_i) is the trajectory reward and d is a diversity function. Our diversity function uses position-based trajectory signatures (rooms visited + final position + path grid cells), which captures finer-grained behavioral differences than room-set-only diversity.

**3. Polychromic Advantage Replacement (Algorithm 2 lines 18–24)**
At rollout states where diversity > 0, the standard GAE advantage is replaced with the polychromic advantage: A_poly = max(scores) - baseline. This is applied to the rollout state and the next W=5 timesteps (the polychromic window). When diversity = 0 (all vine trajectories follow identical paths), GAE is preserved to maintain learning signal. Only the actor's advantage is replaced — the critic's return target remains the original GAE return.

**4. KL Penalty (Algorithm 2 line 31)**
A KL divergence penalty KL(π_β ∥ π_θ) anchors the policy near the behavior policy, preventing exploration from destabilizing training. The behavior policy is updated after each epoch (Algorithm 2 line 33).

### Key Design Decisions

- **BFS expert demos**: Equivalent to the paper's Minari dataset — both provide optimal shortest-path demonstrations. Our BFS generates 5000 episodes (84K transitions) vs Minari's 590 episodes (10K steps).
- **Shared ActorCritic**: The paper says "compact MLP" without specifying shared vs separate. We use shared (147→64→64→heads) because the small network benefits from shared feature extraction.
- **Position-based diversity**: The paper uses room-based diversity for FourRooms. We extend to position-based signatures because room-based diversity is too coarse — many vine rollouts visit the same rooms via the same path, yielding diversity=0 and eliminating the polychromic signal.
- **Checkpoint by eval success**: Pretraining selects the epoch with highest eval success rate (not validation accuracy), since a model with 88% per-step accuracy can still fail to reach the goal if it errs at critical junctions.

---

## Gap Analysis

### Why PPO exceeds the paper

Our PPO (91.0%) slightly exceeds the paper's PPO (89.2%). This is likely due to our use of 50 parallel environments (one per config) which provides more diverse training signal per iteration, and 10 update epochs which enables thorough learning from each batch.

### Why REINFORCE falls short (70.3% vs 89.6%)

The paper does not publish REINFORCE hyperparameters (learning rate, batch size, whether GAE or pure MC returns are used). Our implementation uses batch rollout with Monte Carlo returns (GAE λ=1.0) and a single gradient step per iteration — the textbook REINFORCE with baseline. Without clipping, training is inherently unstable: success rates oscillate ±20% across evaluations. This instability is the fundamental limitation that motivated PPO's development.

### Why Poly-PPO < PPO (83.0% vs 91.0%)

The paper shows Poly-PPO (92.4%) > PPO (89.2%). In our results, Poly-PPO (83.0%) < PPO (91.0%). This gap has two root causes:

**1. Undisclosed architecture details.** Table 3 hyperparameters are shared across all environments, including BabyAI tasks that use CNN-GRU with far more parameters. The paper's compact MLP for FourRooms is not fully specified (hidden_dim, layers, activation, shared vs separate). We verified experimentally that Table 3's exact values (lr=1e-5, epochs=2, gamma=1.0) produce only 71–77% on our architecture, necessitating adaptation.

**2. Sparse diversity signal.** The polychromic objective only provides gradient signal when vine trajectories exhibit diverse behavior. In FourRooms with a well-trained policy, many rollouts follow similar optimal paths, causing diversity≈0 at many rollout states. The polychromic advantage then falls back to standard GAE, and Poly-PPO effectively becomes a PPO variant with fewer update epochs.

The paper's Poly-PPO advantage is most pronounced in pass@k experiments (Table 2), which measure whether diverse strategies exist across multiple attempts — a metric that rewards trajectory-level diversity rather than single-attempt success rate.

---

## Paper Reference

```bibtex
@article{hamid2026polychromic,
    title={Polychromic Objectives for Reinforcement Learning},
    author={Hamid, Fahim Tajwar and others},
    journal={arXiv preprint arXiv:2509.25424v3},
    year={2026}
}
```
