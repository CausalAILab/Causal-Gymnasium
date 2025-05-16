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
        epsilon: float = 0.1, # Epsilon for epsilon-greedy exploration in act()
        seed: int = 0 # Seed for UCBVI's internal randomness
    ):
        # Number of states, actions, and planning horizon
        self.S = num_states
        self.A = n_actions
        self.H = horizon
        self.delta = delta
        self.epsilon = epsilon # Store epsilon
        self.np_random = np.random.RandomState(seed) # Initialize own random number generator
        # Ensure delta is not zero or too small to prevent log(0) or extreme log_inv_delta
        if self.delta <= 1e-9: # Threshold for practical purposes
            print(f"[Warning] UCBVI delta is very small ({self.delta}). Adjusting to 1e-9 to avoid issues.")
            self.delta = 1e-9
        self.log_inv_delta = np.log(1.0 / self.delta)
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

    def update(self, s: int, x_int: int, a: int, r: float, s_next: int, print_diagnostics: bool = False):
        """
        Update rewards, counts, and transition estimate for tuple (s, intended, applied) -> s_next.
        """
        # Store old R for comparison if reward is positive
        old_R_val = self.R[s, x_int, a]
        
        self.N[s, x_int, a] += 1
        alpha = 1.0 / self.N[s, x_int, a]
        # exponential moving average of reward
        self.R[s, x_int, a] = (1 - alpha) * self.R[s, x_int, a] + alpha * r
        # update transition counts and renormalize
        self.P[s, x_int, a, s_next] = (1 - alpha) * self.P[s, x_int, a, s_next] + alpha
        
        # Avoid division by zero if sum is zero, though P should eventually sum to 1
        current_P_sum = self.P[s, x_int, a, :].sum()
        if current_P_sum > 1e-9: # Check if sum is not too close to zero
            self.P[s, x_int, a, :] /= current_P_sum
        else:
            # If sum is zero (e.g. first update and P_s_x_a_s_next was also zero before alpha),
            # and alpha made P[s,x_int,a,s_next] non-zero, this path might be taken.
            # This case should be rare if P is initialized to 0 and alpha > 0.
            # For a deterministic update, P[s,x_int,a,s_next] becomes 1.
            # If for some reason P[s,x_int,a,:] sums to 0 after the update, log it.
            # This implies P[s,x_int,a,s_next] was 0, and alpha might also be 0 if N is huge, or P update logic is flawed.
            # With N starting at 1, alpha = 1/(N_old+1).
            # If P[s,x_int,a,s_next] was 0, it becomes alpha. The sum becomes alpha. So P[s,x_int,a,s_next] normalized is 1.
            # This "else" branch for current_P_sum being zero should ideally not be hit with N starting at 1.
            pass

        if print_diagnostics or r > 0 or s == 14 or s_next == 15:
            print(f"  UCBVI.update: s={s}, x_int={x_int}, a_exec={a}, r={r:.2f}, s'={s_next}")
            print(f"    N[{s},{x_int},{a}]={self.N[s,x_int,a]-1}->{self.N[s,x_int,a]}, R[{s},{x_int},{a}]={old_R_val:.2f}->{self.R[s,x_int,a]:.2f}")
            if s == 14 and a == 2: # Action RIGHT from state before goal
                 print(f"    P[14,{x_int},2,{s_next}] updated. P(15|14,{x_int},2)={self.P[14,x_int,2,15]:.2f}")

    def plan(self, num_sweeps: int | None = None, print_diagnostics: bool = False, env_desc=None):
        sweeps = num_sweeps or self.H
        """
        Perform H iterations of backward optimistic value iteration:
        Q = R + bonus + P * V
        V = max_a Q
        """
        for _ in range(sweeps):
            # compute optimism bonus
            bonus = self.bonus()  # shape (S, A, A)
            
            # Calculate V_star[s'] = max_{x_int'} V[s', x_int']
            # self.V has shape (S, A) where second axis is x_int
            V_star_next_state = np.max(self.V, axis=1) # Shape (S)

            # expected next-value under each (s, x_int, a)
            # P has shape (S, A, A, S) -> (s, x_int, a, s_next)
            # V_star_next_state has shape (S) -> (s_next)
            # expected = np.einsum('sxas,sx->sxa', self.P, self.V) # Original
            expected = np.einsum('sxas,s->sxa', self.P, V_star_next_state) # Modified
            
            # update optimistic Q
            self.Q = self.R + bonus + expected
            # update V by taking max over applied action
            self.V = np.max(self.Q, axis=2)

            # Explicitly set V=0 for terminal states (Hole or Goal)
            if env_desc is not None:
                # Assuming env_desc is a 2D numpy array (like FrozenLake desc)
                # Need to know ncol for _to_rc conversion if not directly accessible
                # For simplicity, assuming S is nrow * ncol and desc has shape (nrow, ncol)
                # This part needs to be robust if SCM can have arbitrary S not tied to grid.
                # For FrozenLake, S = nrow * ncol.
                try:
                    nrow, ncol = env_desc.shape
                    if self.S == nrow * ncol: # Basic check for grid-like env
                        for s_idx in range(self.S):
                            r, c = s_idx // ncol, s_idx % ncol
                            cell_char = env_desc[r,c].item()
                            if isinstance(cell_char, bytes):
                                cell_char = cell_char.decode('utf-8')
                            if cell_char == 'H' or cell_char == 'G':
                                self.V[s_idx, :] = 0.0
                    # else: 
                        # print("[Warning] UCBVI.plan: env_desc shape mismatch with S, not setting terminal V=0.")
                except AttributeError: # env_desc might not have .shape (e.g. if not a numpy array)
                    # print("[Warning] UCBVI.plan: env_desc has no shape, not setting terminal V=0.")
                    pass # Or handle other desc types
            
            # Clip V to be within reasonable bounds. Max possible sum of rewards is H if max_r=1, min is -H if min_r=-1.
            self.V = np.clip(self.V, -self.H, self.H)

        if print_diagnostics:
            print(f"  UCBVI.plan end (sweeps={sweeps}):")
            if self.S > 15: # Ensure states 0, 14, 15 exist
                print(f"    V[0,:]={np.round(self.V[0,:], 3)}")
                print(f"    V[14,:]={np.round(self.V[14,:], 3)}")
                print(f"    V[15,:]={np.round(self.V[15,:], 3)}") # Goal state V (expect 0 if terminal)
                # Q[s,x_int,a]. Q for state 14, for each intended action x_int_diag, applying action RIGHT (2)
                for x_int_diag in range(self.A):
                    print(f"    Q[14,{x_int_diag},2]={self.Q[14,x_int_diag,2]:.3f} (s14, intend {x_int_diag}, applied RIGHT to G)")
            else: # Smaller map, print just V[0]
                print(f"    V[0,:]={np.round(self.V[0,:], 3)}")

    def random_argmax(self, value_array):
        """Helper function to randomly select among max values."""
        if not isinstance(value_array, np.ndarray):
            value_array = np.array(value_array)
        max_val = np.max(value_array)
        return np.random.choice(np.where(value_array == max_val)[0])

    def act(self, s: int, x_int: int) -> int:
        """
        Return the action a that maximizes optimistic Q in state s with intended action x_int,
        breaking ties randomly, with epsilon-greedy exploration.
        """
        if self.np_random.random() < self.epsilon:
            return self.np_random.choice(self.A) # Return a random action
        else:
            return int(self.random_argmax(self.Q[s, x_int])) # Greedy action

    def reset_model(self):
        """
        Reset all estimates (counts, rewards, transitions, Q, V) to initial state.
        """
        self.Q.fill(0)
        self.V.fill(0)
        self.N.fill(1)
        self.R.fill(0)
        self.P.fill(0)
