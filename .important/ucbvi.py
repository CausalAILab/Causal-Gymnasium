# =============================================================
# ucbvi.py  — Algorithm 26: Counterfactual‑UCB driver (outer loop)
# =============================================================
"""
Implements the episode loop described in Algorithm 26.
It keeps the empirical model (counts & rewards) and calls `ucb_q` (Alg‑27)
before each episode to refresh the optimistic tables.

Public API
----------
observe(...):    add one transition datum (s,x,a,r,s').
plan():          refresh `Q` & `V` tables using current counts.
act(s,x_int):    return greedy override action  argmax_a Q[s,x,a].
"""

import numpy as np
from typing import Optional

# Import the builder from the same package
from ucbq import ucb_q


class CtfUCBDriver:
    """Outer‑loop Counterfactual UCB (Alg‑26)."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        horizon: int, # Number of planning sweeps (integer)
        max_episode_reward: float, # Max possible reward in an episode
        delta: float = 0.05,
        c_bonus: float = 1.0, # Defaulting to 1.0 as it showed promise
        verbose: bool = False,
    ) -> None:
        self.S, self.X, self.A = n_states, n_actions, n_actions
        self.H_planning_sweeps = horizon # Store as planning sweeps horizon
        self.max_r = max_episode_reward  # Store max episodic reward
        self.delta, self.c = delta, c_bonus
        self.verbose = verbose

        # empirical model
        # s_n_counts[s, x, a, s_next] = count of (s,x,a) transitioning to s_next
        self.s_n_counts = np.zeros((self.S, self.X, self.A, self.S), dtype=np.int32)
        # R_sum[s, x, a] = sum of rewards received after (s,x,a)
        self.R_sum = np.zeros((self.S, self.X, self.A), dtype=np.float32)
        # N_visits[s, x, a] = count of (s,x,a) occurrences
        self.N_visits = np.zeros((self.S, self.X, self.A), dtype=np.int32)

        # optimistic tables  (filled by plan())
        self.Q = np.zeros((self.S, self.X, self.A), dtype=np.float32)
        self.V = np.zeros((self.S, self.X), dtype=np.float32)

    # ------------- data collection -----------------------------
    def observe(
        self,
        s: int,
        x_intent: int,
        a_override: int,
        r: float,
        s_next: int,
    ) -> None:
        """Log one transition (Step 5 of Alg‑26)."""
        if self.verbose:
            print(f"  UCBVI.observe: s={s}, x_intent={x_intent}, a_override={a_override}, r={r:.2f}, s_next={s_next}")

        self.N_visits[s, x_intent, a_override] += 1
        self.R_sum[s, x_intent, a_override] += r
        self.s_n_counts[s, x_intent, a_override, s_next] += 1
        
        if self.verbose:
            print(f"    Updated counts: N_visits[{s},{x_intent},{a_override}] = {self.N_visits[s, x_intent, a_override]}")
            print(f"    Updated R_sum[{s},{x_intent},{a_override}] = {self.R_sum[s, x_intent, a_override]:.2f}")
            print(f"    Updated s_n_counts[{s},{x_intent},{a_override},{s_next}] = {self.s_n_counts[s, x_intent, a_override, s_next]}")

    # ------------- planning ------------------------------------
    def plan(self) -> None:
        """Step 7 of Alg‑26 – call Alg‑27 to refresh optimistic tables."""
        if self.verbose:
            print(f"\nUCBVI.plan: Starting planning cycle. Planning_sweeps={self.H_planning_sweeps}, Max_episode_reward={self.max_r}")
            print(f"  Shapes: N_visits={self.N_visits.shape}, R_sum={self.R_sum.shape}, s_n_counts={self.s_n_counts.shape}")
            
        self.Q, self.V = ucb_q(
            s_n_counts=self.s_n_counts,
            R_sum=self.R_sum,
            N_visits=self.N_visits,
            horizon=self.H_planning_sweeps, # Pass planning sweeps horizon (int)
            max_episode_reward=self.max_r,  # Pass max episodic reward (float)
            delta=self.delta,
            c_bonus=self.c,
            verbose=self.verbose
        )
        if self.verbose:
            if self.S > 0:
                print(f"  Q-values for S=0 after planning (X rows, A columns):\n{self.Q[0, :, :]}")
            print(f"UCBVI.plan: Finished planning cycle.\n")

    # ------------- acting --------------------------------------
    def act(self, s: int, x_intent: int) -> int:
        """Greedy override action argmax_a Q[s,x,a] with random tie‑break."""
        if self.verbose:
            print(f"  UCBVI.act: s={s}, x_intent={x_intent}")
            if self.S > s and self.X > x_intent: # Check bounds
                print(f"    Q-values Q[{s},{x_intent},:]: {self.Q[s, x_intent, :]}")
        
        q_row = self.Q[s, x_intent]
        max_q = np.max(q_row) # Use np.max for robustness if q_row can be all -inf or nan
        
        # Get all indices where q_row equals max_q
        best_actions = np.flatnonzero(q_row == max_q)
        
        if len(best_actions) == 0: # Should not happen if Q is initialized (e.g. to 0)
            if self.verbose:
                print(f"    WARN: No best action found for Q[{s},{x_intent},:], defaulting to action 0.")
            chosen_action = 0
        else:
            chosen_action = int(np.random.choice(best_actions))

        if self.verbose:
            print(f"    Chosen action a: {chosen_action}")
        return chosen_action

