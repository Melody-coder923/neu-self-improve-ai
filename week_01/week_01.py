import numpy as np
from math import comb
import matplotlib.pyplot as plt

# ====================================================
# Hotel Overbooking Problem (MDP)
# Policy Iteration — Finite Horizon
# ====================================================

# Parameters
CAPACITY = 100
MAX_OVERBOOK = 20
MAX_ROOMS = CAPACITY + MAX_OVERBOOK  # max bookings we can hold
DAYS = 30

ROOM_PRICE = 100
PENALTY = 250   # cost per overbooked guest

P_ARRIVAL = 0.8
P_CANCEL = 0.01
P_NOSHOW = 0.05

GAMMA = 1.0  # finite horizon, no discounting needed

# -------------------------
# Helper functions
# -------------------------
def binom_prob(n, k, p):
    # Binomial: P(X=k) for X~Bin(n,p)
    if k < 0 or k > n:
        return 0.0
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def terminal_reward(n):
    # Check-in day profit: each guest shows up with prob (1-P_NOSHOW)
    r = 0.0
    for actual in range(n + 1):
        p = binom_prob(n, actual, 1 - P_NOSHOW)
        served = min(actual, CAPACITY)
        over = max(0, actual - CAPACITY)
        r += p * (served * ROOM_PRICE - over * PENALTY)
    return r

def transitions(n, action):
    # Returns {next_state: probability}
    # Cancel first, then arrival (new booking can't cancel same day)

    # cancellations on existing bookings
    after_cancel = {}
    for c in range(n + 1):
        p_c = binom_prob(n, c, P_CANCEL)
        if p_c < 1e-12:
            continue
        after_cancel[n - c] = after_cancel.get(n - c, 0.0) + p_c

    # arrival (only if accepted AND room available)
    final = {}
    for n_mid, p_mid in after_cancel.items():
        if action == 1 and n_mid < MAX_ROOMS:
            n_next = n_mid + 1
            final[n_next] = final.get(n_next, 0.0) + p_mid * P_ARRIVAL
            final[n_mid]  = final.get(n_mid, 0.0)  + p_mid * (1 - P_ARRIVAL)
        else:
            final[n_mid] = final.get(n_mid, 0.0) + p_mid
    return final

# -------------------------
# Initialization
# -------------------------
V = np.zeros((DAYS + 1, MAX_ROOMS + 1))
policy = np.zeros((DAYS + 1, MAX_ROOMS + 1), dtype=int)

# base case: check-in day
for n in range(MAX_ROOMS + 1):
    V[0, n] = terminal_reward(n)

# -------------------------
# Policy Iteration
# -------------------------
def policy_eval():
    # sweep forward: V[t] uses V[t-1] which is already computed
    for t in range(1, DAYS + 1):
        for n in range(MAX_ROOMS + 1):
            a = policy[t, n]
            V[t, n] = GAMMA * sum(p * V[t-1, nxt]
                                  for nxt, p in transitions(n, a).items())

def policy_improve():
    stable = True
    for t in range(1, DAYS + 1):
        for n in range(MAX_ROOMS + 1):
            old = policy[t, n]

            q = [GAMMA * sum(p * V[t-1, nxt]
                             for nxt, p in transitions(n, a).items())
                 for a in [0, 1]]

            if n >= MAX_ROOMS:
                policy[t, n] = 0  # already full
            else:
                policy[t, n] = int(q[1] > q[0])

            if policy[t, n] != old:
                stable = False
    return stable

iters = 0
for _ in range(50):
    iters += 1
    policy_eval()
    if policy_improve():
        break

# -------------------------
# Output
# -------------------------

# find booking threshold for each day
threshold = np.zeros(DAYS + 1, dtype=int)
for t in range(1, DAYS + 1):
    for n in range(MAX_ROOMS + 1):
        if policy[t, n] == 0:
            threshold[t] = n
            break
    else:
        threshold[t] = MAX_ROOMS

# print sample
sample_n = [80, 90, 100, 105, 110, 115, 120]
sample_t = [30, 20, 10, 5, 1]

print(f"Policy Iteration converged in {iters} iterations.\n")
print("Optimal Policy (sample):")
header = "Days Left | " + " | ".join(f"{n:>3} booked" for n in sample_n)
print(header)
print("-" * len(header))
for t in sample_t:
    row = ["Accept" if policy[t, n] == 1 else "Reject" for n in sample_n]
    print(f"    {t:>5} | " + " | ".join(f"{x:>9}" for x in row))

# plot
plt.figure(figsize=(8, 5))
plt.step(range(1, DAYS + 1), threshold[1:], where="mid", linewidth=2,
         label="Booking threshold $n^*$")
plt.axhline(CAPACITY, color="red", linestyle="--", linewidth=1.5,
            label=f"Capacity ({CAPACITY})")
plt.xlabel("Days until check-in")
plt.ylabel("Booking threshold $n^*$")
plt.title("Optimal Overbooking Threshold")
plt.gca().invert_xaxis()
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("optimal_threshold.png", dpi=150)

print(f"\nExpected revenue (start empty, t={DAYS}): ${V[DAYS, 0]:,.2f}")
print("Plot saved to optimal_threshold.png")
