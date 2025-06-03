# =============================================================
# ucbvi.py  — Algorithm 26: Counterfactual‑UCB driver (outer loop)
# =============================================================
"""
Implements the episode loop described in Algorithm 26.
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
        horizon: int,
        delta: float = 0.05,
        c_bonus: float = 7.0,
    ) -> None:
        self.S, self.X, self.A, self.H = n_states, n_actions, n_actions, horizon
        self.delta, self.c = delta, c_bonus

        # empirical model
        self.C = np.zeros((self.S, self.X, self.A, self.S), dtype=np.int32)
        self.R = np.zeros((self.S, self.X, self.A), dtype=np.float32)

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
        """Log one transition (Step 5 of Alg‑26)."""
        # counts
        self.C[s, x_intent, a_override, s_next] += 1

        # incremental reward mean
        N = self.C[s, x_intent, a_override].sum()
        self.R[s, x_intent, a_override] += (
            r - self.R[s, x_intent, a_override]
        ) / N

    # ------------- planning ------------------------------------
    def plan(self) -> None:
        """Step 7 of Alg‑26 – call Alg‑27 to refresh optimistic tables."""
        self.Q, self.V = ucb_q(
            self.C, self.R, self.H, delta=self.delta, c_bonus=self.c
        )

    # ------------- acting --------------------------------------
    def act(self, s: int, x_intent: int) -> int:
        """Greedy override action argmax_a Q[s,x,a] with random tie‑break."""
        q_row = self.Q[s, x_intent]
        max_q = q_row.max()
        return int(np.random.choice(np.flatnonzero(q_row == max_q)))

