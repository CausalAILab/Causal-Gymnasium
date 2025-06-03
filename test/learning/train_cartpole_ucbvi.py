import argparse
import os
import sys
import time
from typing import List, Tuple, Any

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Adjust sys.path
script_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_file_path) # causalgym/test/learning/
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..")) # C:/Users/Matthew/Documents/causal2/
algorithms_dir = os.path.join(project_root, "causalgym", "causal_gym", "algorithms")

if project_root not in sys.path:
    sys.path.insert(0, project_root) # For cartpole_wind_pch.py (if in root)
if algorithms_dir not in sys.path:
    sys.path.insert(0, algorithms_dir) # For ucbvi.py, ucbq.py

try:
    from causal_gym.envs.cartpole_wind import CartPoleWindPCH
    from ucbvi import UCBVI # Assumed in algorithms_dir
    try:
        from ucbq import UCBQ # Assumed in algorithms_dir
    except ImportError:
        print("Warning: ucbq.py not found or UCBQ class not importable. --algo ucbq will not be available.")
        UCBQ = None
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print(f"Looked in sys.path including: {project_root} and {algorithms_dir}")
    print("Please ensure 'cartpole_wind_pch.py' is in the project root,")
    print("and 'ucbvi.py' / 'ucbq.py' are in 'causalgym/causal_gym/algorithms/'.")
    sys.exit(1)


class Discretiser:
    """
    Discretises continuous observation space for CartPoleWindPCH into integer states.
    Uses symmetric ranges:
      position ∈ [-2.4, 2.4], velocity ∈ [-3.0, 3.0],
      angle ∈ [-0.20, 0.20], angular-vel ∈ [-3.5, 3.5].
    """
    def __init__(self, bins_per_dim: int):
        if not isinstance(bins_per_dim, int) or bins_per_dim <= 0:
            raise ValueError("bins_per_dim must be a positive integer.")
        self.bins_per_dim_each = np.array([bins_per_dim] * 4, dtype=np.int_) # bins for each of 4 dims

        self.obs_low = np.array([-2.4, -3.0, -0.20, -3.5], dtype=np.float32)
        self.obs_high = np.array([2.4, 3.0, 0.20, 3.5], dtype=np.float32)
        self.num_dims = 4

        self.bin_edges: List[np.ndarray] = []
        for i in range(self.num_dims):
            # Create bins + 1 edges to get 'bins' segments
            edges = np.linspace(self.obs_low[i], self.obs_high[i], self.bins_per_dim_each[i] + 1)
            self.bin_edges.append(edges)

        self._n_states = int(np.prod(self.bins_per_dim_each))
        if self._n_states <= 0:
            raise ValueError("Number of states must be positive.")

    def __call__(self, obs: np.ndarray) -> int:
        bin_indices = np.empty(self.num_dims, dtype=np.int_)
        for i in range(self.num_dims):
            # Clip observation to the defined discretization range
            clipped_obs_i = np.clip(obs[i], self.obs_low[i], self.obs_high[i])
            
            if self.bins_per_dim_each[i] == 1: # Single bin means index is always 0
                bin_indices[i] = 0
            else:
                thresholds = self.bin_edges[i][1:-1] 
                bin_indices[i] = np.digitize(clipped_obs_i, thresholds)
        
        state_index = 0
        for i in range(self.num_dims):
            state_index = state_index * self.bins_per_dim_each[i] + bin_indices[i]
        return int(state_index)

    @property
    def n_states(self) -> int:
        return self._n_states

def main(args: argparse.Namespace):
    np.random.seed(args.seed)

    env = CartPoleWindPCH() 

    discretiser = Discretiser(bins_per_dim=args.bins)
    print(f"Discretised states: {discretiser.n_states} (bins per dim: {args.bins})")

    n_actions = env.action_space.n 
    if args.algo == "ucbvi":
        agent = UCBVI(
            num_states=discretiser.n_states,
            n_actions=n_actions,
            horizon=args.horizon,
            delta=args.delta,
            epsilon=args.epsilon, 
            seed=args.seed
        )
    elif args.algo == "ucbq":
        if UCBQ is None:
            print("Error: UCBQ algorithm selected, but ucbq.py could not be imported or UCBQ class is not available.")
            sys.exit(1)
        agent = UCBQ(
            n_states=discretiser.n_states,
            n_actions=n_actions,
            delta=args.delta
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    episode_returns: List[float] = []
    episode_regrets: List[float] = [] 

    print(f"Starting training with {args.algo.upper()} for {args.episodes} episodes...")
    start_time = time.time()

    for ep in range(args.episodes):
        obs_continuous, _ = env.reset(seed=args.seed + ep)
        s = discretiser(obs_continuous)
        ep_return = 0.0
        ep_regret_steps = 0.0 

        if hasattr(agent, 'plan'): 
            agent.plan()

        for t in range(args.horizon):
            if args.algo == "ucbvi":
                x_intended = np.argmax(agent.V[s, :])
                a_to_apply = agent.act(s, x_intended)
            elif args.algo == "ucbq":
                a_to_apply = agent.act(s)
            else: 
                a_to_apply = env.action_space.sample()

            obs2_continuous, r, terminated, truncated, info = env.do(a_to_apply)
            s2 = discretiser(obs2_continuous)
            done = terminated or truncated
            
            realised_action = info.get("realised_action", a_to_apply) 

            if args.algo == "ucbvi":
                agent.update(s, x_intended, realised_action, r, s2)
            elif args.algo == "ucbq":
                agent.update(s, realised_action, r, s2)
            
            ep_return += r
            ep_regret_steps += (1.0 - r) 
            s = s2
            
            if done:
                break
        
        episode_returns.append(ep_return)
        episode_regrets.append(ep_regret_steps) 

        if (ep + 1) % max(1, args.episodes // 10) == 0 or ep == args.episodes - 1:
            print(f"Ep {ep+1:4d}/{args.episodes} | Return: {ep_return:7.2f} | Steps: {t+1} | Regret this ep: {ep_regret_steps:.2f}")
    
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f} seconds.")

    if args.plot:
        plot_filename = "cartpole_regret.png"
        print(f"Saving regret plot to {plot_filename}...")
        plt.figure(figsize=(10, 6))
        cumulative_regret = np.cumsum(episode_regrets)
        plt.plot(cumulative_regret)
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Regret (Sum of (1 - reward_step))")
        plt.title(f"Cumulative Regret for {args.algo.upper()} on CartPoleWind (Bins: {args.bins})")
        plt.grid(True)
        plt.savefig(plot_filename) # Saves in the script's CWD, which will be causalgym/test/learning/
        plt.close()
        print(f"Saved {plot_filename}")

    if args.gif:
        gif_filename = "cartpole.gif"
        print(f"Generating GIF: {gif_filename}...")
        gif_env = CartPoleWindPCH(render_mode="rgb_array")
        frames: List[np.ndarray] = []
        
        obs_gif_cont, _ = gif_env.reset(seed=args.seed + args.episodes + 1) 
        s_gif = discretiser(obs_gif_cont)
        gif_ep_return = 0.0

        if hasattr(agent, 'plan'): 
            agent.plan() 

        for t_gif in range(args.horizon):
            frame = gif_env.render() 
            if frame is None:
                print("Warning: gif_env.render() returned None. Skipping frame.")
                break 
            frames.append(frame)
            
            if args.algo == "ucbvi":
                x_intended_gif = np.argmax(agent.V[s_gif, :])
                a_gif = agent.act(s_gif, x_intended_gif)
            elif args.algo == "ucbq":
                a_gif = agent.act(s_gif)
            else:
                a_gif = gif_env.action_space.sample()
            
            obs2_gif_cont, r_gif, term_gif, trunc_gif, _ = gif_env.do(a_gif)
            s2_gif = discretiser(obs2_gif_cont)
            s_gif = s2_gif
            gif_ep_return += r_gif
            
            if term_gif or trunc_gif:
                break
        
        gif_env.close()
        print(f"GIF rollout return: {gif_ep_return:.2f}, Steps: {t_gif+1}")

        if frames:
            fig_gif = plt.figure()
            plt.axis('off')
            patch = plt.imshow(frames[0])
            
            def animate_fn(i):
                patch.set_data(frames[i])
                return patch,

            anim = animation.FuncAnimation(fig_gif, animate_fn, frames=len(frames), interval=50, blit=True)
            try:
                anim.save(gif_filename, writer="pillow", fps=20) # Saves in CWD
                print(f"Saved {gif_filename}")
            except Exception as e_gif:
                print(f"Error saving GIF: {e_gif}. Is Pillow installed? (`pip install Pillow`)")
            plt.close(fig_gif)
        else:
            print("No frames collected for GIF. Ensure render_mode='rgb_array' is working.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UCB-VI or UCB-Q on CartPoleWindPCH with discretised states.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--horizon", type=int, default=200, help="Maximum steps per episode (planning horizon for UCBVI).")
    parser.add_argument("--algo", type=str, default="ucbvi", choices=["ucbvi", "ucbq"], help="Algorithm to use.")
    parser.add_argument("--bins", type=int, default=8, help="Number of bins per dimension for discretiser.")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta parameter for UCB algorithms.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-greedy exploration (used by UCBVI).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--gif", action="store_true", help="Save a GIF of a trained policy rollout to cartpole.gif.")
    parser.add_argument("--plot", action="store_true", help="Save a PNG plot of cumulative regret to cartpole_regret.png.")
    
    print("Running Discretiser self-test...")
    _test_bins = 8
    _d_test = Discretiser(bins_per_dim=_test_bins)
    assert _d_test.n_states == _test_bins**4, f"Test failed: n_states={_d_test.n_states}, expected={_test_bins**4}"
    _test_obs = np.array([0.0, 0.0, 0.0, 0.0]) 
    _test_idx = _d_test(_test_obs)
    assert 0 <= _test_idx < _d_test.n_states, f"Test failed: idx={_test_idx} out of range for n_states={_d_test.n_states}"
    print("Discretiser self-test passed.")

    parsed_args = parser.parse_args()
    main(parsed_args) 