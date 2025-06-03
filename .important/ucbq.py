# ============================================
# ucbq.py  — Algorithm 27: Optimistic Q builder
# ============================================
"""
Given empirical counts & rewards for every triple (s, x, a, s'), compute the
optimistic Q‑table and the associated V‑table that Counterfactual‑UCB (Alg‑26)
will use for action selection.

Args
----
C : np.ndarray  shape (S, X, A, S)
      Raw transition counts  C[s,x,a,s'].
R : np.ndarray  shape (S, X, A)
      Running mean reward  R̄[s,x,a].
H : int         Horizon (upper bound on cumulative reward).
delta : float    Failure‑probability parameter in the confidence bounds.
c_bonus : float  Exploration constant (≈7 from theory).

Returns
-------
Q : np.ndarray  shape (S, X, A)  — optimistic action‑value table.
V : np.ndarray  shape (S, X)     — optimistic state‑value table  max_a Q.
"""

from typing import Tuple
import numpy as np

PRINT_S_X_A_TUPLE = (0, 0, 0) # Example: State 0, Wind 0, Action 0 for detailed print

def ucb_q(
    s_n_counts: np.ndarray, 
    R_sum: np.ndarray,      
    N_visits: np.ndarray,   
    horizon: int,           # Number of planning sweeps (integer)
    max_episode_reward: float, # Max possible episodic reward for clipping/bonus scaling
    delta: float = 0.05,
    c_bonus: float = 1.0,   # Defaulting to 1.0 as it showed promise
    verbose: bool = False   
) -> Tuple[np.ndarray, np.ndarray]:
    S, X, A, _ = s_n_counts.shape 
    log_term = np.log(1.0 / delta)

    if verbose:
        print(f"    ucbq: Starting. Planning_sweeps(horizon)={horizon}, Max_reward_for_bonus_clip={max_episode_reward}, Delta={delta}, C_bonus={c_bonus}")
        print(f"    ucbq: Shapes: s_n_counts={s_n_counts.shape}, R_sum={R_sum.shape}, N_visits={N_visits.shape}")

    # N_sxa is N(s,x,a) - total visits to (s,x,a) state-intervention-action tuple
    N_sxa = np.maximum(N_visits, 1)  # Avoid divide‑by‑zero for bonus and r_estimate
    N_sxa_for_P = N_visits # For P_hat, use actual N_visits, can be 0 if not visited
    denom_P = np.maximum(N_sxa_for_P, 1) # Avoid divide-by-zero for P_hat calculation

    # r_estimate[s,x,a] = R_sum[s,x,a] / N_sxa[s,x,a]
    r_estimate = R_sum / N_sxa

    # P_hat[s,x,a,s_next] = s_n_counts[s,x,a,s_next] / denom_P[s,x,a]
    P_hat = s_n_counts / denom_P[..., None] # ... and add new dimension for s_next

    # bonuses b[s,x,a] - now scaled by max_episode_reward
    bonus = c_bonus * max_episode_reward * np.sqrt(log_term / N_sxa)

    V_optimistic = np.zeros((S, X), dtype=np.float32)
    Q_optimistic = np.zeros((S, X, A), dtype=np.float32)

    if verbose and PRINT_S_X_A_TUPLE[0] < S and PRINT_S_X_A_TUPLE[1] < X and PRINT_S_X_A_TUPLE[2] < A:
        s_p, x_p, a_p = PRINT_S_X_A_TUPLE
        print(f"    ucbq: Initial R_sum[{s_p},{x_p},{a_p}]={R_sum[s_p,x_p,a_p]:.2f}, N_visits[{s_p},{x_p},{a_p}]={N_visits[s_p,x_p,a_p]}")
        print(f"    ucbq: Initial r_estimate[{s_p},{x_p},{a_p}]={r_estimate[s_p,x_p,a_p]:.2f}, bonus[{s_p},{x_p},{a_p}]={bonus[s_p,x_p,a_p]:.2f} (using max_r={max_episode_reward})")
        # print(f"    ucbq: P_hat for ({s_p},{x_p},{a_p}) -> {P_hat[s_p,x_p,a_p,:5]}") # Print first 5 s_next probs

    for h_step in range(horizon):
        # V_expect_future: For each current state 's_curr', this is E_{s_next ~ P_hat(s_next | s_curr, x, a)} [V_optimistic(s_next, x_prime_avg)]
        # where x_prime_avg is averaging over the SCM's next random action at s_next.
        # V_optimistic has shape (S, X). V_optimistic.mean(axis=1) averages over X, giving shape (S,)
        # This means we assume the next SCM action x' is uniformly random for future V.
        V_avg_over_next_x_at_s_next = V_optimistic.mean(axis=1)  # Shape (S,)
        
        # future_value_for_sxa = sum_{s_next} P_hat(s_next | s, x, a) * V_avg_over_next_x_at_s_next[s_next]
        future_value_for_sxa = np.einsum("sxas,s->sxa", P_hat, V_avg_over_next_x_at_s_next)
        
        # Q-values are clipped by max_episode_reward
        new_Q_sxa = np.minimum(max_episode_reward, r_estimate + future_value_for_sxa + bonus)
        
        if verbose and h_step % (horizon // 4 + 1) == 0: # Print periodically during sweeps
            if PRINT_S_X_A_TUPLE[0] < S and PRINT_S_X_A_TUPLE[1] < X and PRINT_S_X_A_TUPLE[2] < A:
                s_p, x_p, a_p = PRINT_S_X_A_TUPLE
                print(f"    ucbq sweep h={h_step}: Q[{s_p},{x_p},{a_p}] updated to {new_Q_sxa[s_p,x_p,a_p]:.2f}")
                print(f"      (r_est={r_estimate[s_p,x_p,a_p]:.2f} + fut={future_value_for_sxa[s_p,x_p,a_p]:.2f} + bon={bonus[s_p,x_p,a_p]:.2f}) clipped by {max_episode_reward}")

        Q_optimistic = new_Q_sxa
        V_optimistic = Q_optimistic.max(axis=2)

    if verbose:
        print(f"    ucbq: Finished backward sweeps.")
    return Q_optimistic, V_optimistic

