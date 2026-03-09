# Assignment 1: Hotel Overbooking Management System using MDP

## 1. MDP Formulation (Slide 1)

We model hotel booking as a finite-horizon MDP that decides whether to accept or reject each booking request.

- **State**: (days left until check-in `t`, current bookings `n`)
- **Action**: Accept (1) / Reject (0) the booking request
- **Transition** (see `transitions()`):
  1. Each existing booking cancels independently with probability `P_CANCEL` (binomial)
  2. If we accept and `n < MAX_ROOMS`, a new booking arrives with probability `P_ARRIVAL`
  3. Order matters: new bookings can only cancel starting from the next day
- **Reward** (see `terminal_reward()`):
  - Intermediate days: **zero** immediate reward
  - Check-in day (t=0): each booked guest shows up w.p. `1 - P_NOSHOW`
    - Revenue: `ROOM_PRICE × min(show-ups, CAPACITY)`
    - Penalty: `PENALTY × max(show-ups - CAPACITY, 0)`
- **Discount factor**: γ = 1 (finite horizon, no discounting needed)

**Bellman Equation:**

$$V(t, n) = \max_a \sum_{n'} P(n' \mid n, a)\, V(t-1, n')$$

Boundary: $V(0, n) = \mathbb{E}[R_{\text{terminal}}(n)]$ computed by `terminal_reward(n)`.

The process satisfies the Markov property: future bookings depend only on the current number of bookings and the action taken, not on past history.

## 2. Experimental Results (Slide 2)

### Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Capacity | C | 100 rooms |
| Max Overbooking | K | 20 rooms |
| Room Price | L | $100 |
| Overbooking Penalty | D | $250 |
| Arrival Prob | p | 0.8 |
| Cancellation Prob | q | 0.01 / day |
| No-show Prob | — | 0.05 |

### Policy Iteration Results

The algorithm converged in **5 iterations**.

### Optimal Policy (Sample)

| Days Left | 80 Booked | 90 Booked | 100 Booked | 105 Booked | 110 Booked | 115 Booked | 120 Booked |
|-----------|-----------|-----------|------------|------------|------------|------------|------------|
| 30 | Accept | Accept | Accept | Accept | Accept | Accept | Reject |
| 20 | Accept | Accept | Accept | Accept | Accept | Accept | Reject |
| 10 | Accept | Accept | Accept | Accept | Accept | Reject | Reject |
| 5 | Accept | Accept | Accept | Accept | Reject | Reject | Reject |
| 1 | Accept | Accept | Accept | Reject | Reject | Reject | Reject |

### Analysis

- The hotel accepts aggressively when check-in is far away (cancellations have time to reduce the count), but becomes conservative close to check-in.
- The optimal policy always overbooks beyond the 100-room capacity — the threshold stays above 100 even at t=1, because a 5% no-show rate justifies a few extra bookings.
- Expected total revenue from empty hotel: **$1,978.28**
  - Revenue is realized only once at check-in. The theoretical upper bound is roughly CAPACITY × ROOM_PRICE = $10,000 (before penalties). The optimal policy often accepts beyond capacity because a 5% no-show rate and daily cancellations reduce effective show-ups; expected profit is lower than the upper bound due to overbooking penalties and uncertainty in arrivals/cancellations.
- The threshold plot (`optimal_threshold.png`) shows maximum bookings allowed for each day.

## 3. Code Implementation

Implemented from scratch in `week_01.py` using only NumPy and Matplotlib:

- **Policy Evaluation** (`policy_eval`): single forward sweep computes exact V, since V[t] only depends on V[t-1]
- **Policy Improvement** (`policy_improve`): greedy update picks the better action for each (t, n)
- **Main loop**: iterates eval → improve until policy stabilizes

## 4. References and Data Sources

### Academic References

- Talluri, K. T., & van Ryzin, G. J. (2004). *The Theory and Practice of Revenue Management*. Springer. (Chapter 4: Overbooking)
- Subramanian, J., Stidham Jr, S., & Lautenbacher, C. J. (1999). Airline overbooking with cancellations and no-shows. *Transportation Science*, 33(2), 147–167.

### Data Justification

- **Penalty Ratio (2.5×)**: Industry standards — "walking a guest" includes alternative lodging, transport, and loyalty compensation (Talluri & van Ryzin, 2004).
- **Cancellation Rate (1%/day)**: Aligns with cumulative cancellation rates of 20–30% for leisure hotels (Hospitality Net).
- **No-show Rate (5%)**: Standard baseline in hospitality revenue management simulations.

[Watch the Presentation Video](https://youtu.be/pd5DQCCKe7I)
