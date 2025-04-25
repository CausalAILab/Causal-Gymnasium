import numpy as np

class UCBVI:
    """
    Counterfactual UCB-VI for episodic MDPs (Alg.26 Ctf-UCBVI).
    Maintains optimistic Q-values over (state, intended_action, chosen_action).
    """
    def __init__(
        self,
        num_states: int,
        n_actions: int,
        horizon: int,
        delta: float = 0.1,
    ):
        # Number of states, actions, and planning horizon
        self.S = num_states
        self.A = n_actions
        self.H = horizon
        self.delta = delta
        self.log_inv_delta = np.log(1.0 / delta)
        # Value tables
        # Q[s, x_int, a] = optimistic estimate for taking a when intended action was x_int in state s
        self.Q = np.zeros((self.S, self.A, self.A))
        # V[s, x_int] = max_a Q[s, x_int, a]
        self.V = np.zeros((self.S, self.A))
        # Counts and estimates
        self.N = np.ones((self.S, self.A, self.A))   # visitation counts
        self.R = np.zeros((self.S, self.A, self.A))  # avg reward
        self.P = np.zeros((self.S, self.A, self.A, self.S))  # transition probabilities

    def bonus(self):
        """Optimism bonus matrix of shape (S, A, A)"""
        return np.sqrt(2 * self.log_inv_delta / self.N)

    def update(self, s: int, x_int: int, a: int, r: float, s_next: int):
        """
        Update rewards, counts, and transition estimate for tuple (s, intended, applied) -> s_next.
        """
        self.N[s, x_int, a] += 1
        alpha = 1.0 / self.N[s, x_int, a]
        # exponential moving average of reward
        self.R[s, x_int, a] = (1 - alpha) * self.R[s, x_int, a] + alpha * r
        # update transition counts and renormalize
        self.P[s, x_int, a, s_next] = (1 - alpha) * self.P[s, x_int, a, s_next] + alpha
        self.P[s, x_int, a, :] /= self.P[s, x_int, a, :].sum()

    def plan(self):
        """
        Perform H iterations of backward optimistic value iteration:
        Q = R + bonus + P * V
        V = max_a Q
        """
        for _ in range(self.H):
            # compute optimism bonus
            bonus = self.bonus()  # shape (S, A, A)
            # expected next-value under each (s, x_int, a)
            # V has shape (S, A), where second axis matches x_int
            expected = np.einsum('sxas,sx->sxa', self.P, self.V)
            # update optimistic Q
            self.Q = self.R + bonus + expected
            # update V by taking max over applied action
            self.V = np.max(self.Q, axis=2)

    def act(self, s: int, x_int: int) -> int:
        """
        Return the action a that maximizes optimistic Q in state s with intended action x_int.
        """
        return int(np.argmax(self.Q[s, x_int]))

    def reset_model(self):
        """
        Reset all estimates (counts, rewards, transitions, Q, V) to initial state.
        """
        self.Q.fill(0)
        self.V.fill(0)
        self.N.fill(1)
        self.R.fill(0)
        self.P.fill(0)
