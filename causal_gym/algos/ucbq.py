import numpy as np
import pandas as pd

# UCB-Q is equivalent generally to:
#
# for each observed transition (s, a, r, s'):
#     N[s,a] += 1
#     b = sqrt(2*log(1/delta) / N[s,a])
#     target = r + max_a' Q[s',a'] + b
#     Q[s,a] = (1 - 1/N[s,a]) * Q[s,a] + (1/N[s,a]) * target
#
# Follows the same UCB exploration principle as UCBVI, but in a model-free Q-learning style.

class UCBQ:
    def __init__(self, n_states=100, n_actions=2, delta=0.1):
        """
        Model-free UCB-Q learning.

        :param n_states: Number of discrete states
        :param n_actions: Number of discrete actions
        :param delta: Confidence parameter for UCB bonus
        """
        self.S = n_states
        self.A = n_actions
        self.delta = delta

        # Q-table and visitation counts
        self.Q = np.zeros((self.S, self.A))
        self.N_sa = np.ones((self.S, self.A))  # start at 1 to avoid divide-by-zero

        # Precompute log(1/delta)
        self.log_inv_delta = np.log(1 / self.delta)

    def bonus(self, s, a):
        """
        Exploration bonus for state-action pair (s, a).
        """
        return np.sqrt(2 * self.log_inv_delta / self.N_sa[s, a])

    def update(self, s, a, r, s_next):
        """
        Perform one UCB-Q update given transition (s, a, r, s_next).
        """
        # Increment count
        self.N_sa[s, a] += 1

        # Learning rate alpha = 1 / N(s,a)
        alpha = 1.0 / self.N_sa[s, a]

        # Compute bonus
        b = self.bonus(s, a)

        # TD-style target with max over next Q and bonus
        target = r + np.max(self.Q[s_next]) + b

        # Update Q-value
        self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * target

    def act(self, s):
        """
        Greedy action selection based on current Q-values.
        """
        return int(np.argmax(self.Q[s]))
