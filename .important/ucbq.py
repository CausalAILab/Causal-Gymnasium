# ============================================
# ucbq.py  — Algorithm 27: Optimistic Q builder
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


def ucb_q(
    C: np.ndarray,
    R: np.ndarray,
    H: int,
    delta: float = 0.05,
    c_bonus: float = 7.0,
) -> Tuple[np.ndarray, np.ndarray]:
    S, X, A, _ = C.shape
    log_term = np.log(1.0 / delta)

    # visit totals N[s,x,a]
    N = C.sum(axis=-1)
    denom = np.maximum(N, 1)  # avoid divide‑by‑zero

    # transition probabilities P[s,x,a,s']
    P = C / denom[..., None]

    # bonuses b[s,x,a]
    bonus = c_bonus * H * np.sqrt(log_term / denom)

    # initialise V_{H+1} = 0
    V = np.zeros((S, X), dtype=np.float32)
    Q = np.zeros((S, X, A), dtype=np.float32)

    # backward sweep  h = H, …, 1
    for _ in range(H):
        # Expected future value  E_{s'}[ V(s', ·) ]  then average over next intent
        V_expect = V.mean(axis=1)  # shape (S,)
        future = np.einsum("sxas,s->sxa", P, V_expect)
        Q = np.minimum(H, R + future + bonus)
        V = Q.max(axis=2)

    return Q, V

