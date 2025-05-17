import argparse
import os
import sys
import time
from typing import List, Tuple, Any

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Adjust sys.path to allow imports from the project root and the algorithms directory
script_dir = os.path.dirname(os.path.abspath(__file__)) # causalgym/test/learning/
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..")) # Up three levels to project root

# Add project root to sys.path for `from causal_gym.envs...`
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add the specific algorithms directory to sys.path for `from ucbvi import UCBVI`
# This assumes ucbvi.py and ucbq.py are directly in that folder and act as modules.
algorithms_path = os.path.join(project_root, "causalgym", "causal_gym", "algorithms")
if algorithms_path not in sys.path:
    sys.path.insert(0, algorithms_path)

# Attempt to import necessary modules
try:
    from causal_gym.envs.lunar_lander import LunarLanderPCH
    from ucbvi import UCBVI
    try:
        from ucbq import UCBQ
    except ImportError:
        UCBQ = None 
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure 'causal_gym' is installed or accessible in the project root,")
    print("and 'ucbvi.py' (and optionally 'ucbq.py', 'lunar_lander.py' if not in causal_gym path) are in the script's directory.")
    sys.exit(1)


class Discretiser:
    """
    Discretises continuous observation space for LunarLander-v2 into integer states.
    """
    def __init__(self, obs_low: np.ndarray, obs_high: np.ndarray, bins_per_dim: List[int]):
        self.obs_low = obs_low
        self.obs_high = obs_high
        self.bins_per_dim = np.array(bins_per_dim, dtype=np.int_)
        self.num_dims = len(bins_per_dim)

        self.bin_edges: List[np.ndarray] = []
        for i in range(self.num_dims):
            edges = np.linspace(self.obs_low[i], self.obs_high[i], self.bins_per_dim[i] + 1)
            self.bin_edges.append(edges)

        self._n_states = np.prod(self.bins_per_dim)
        if self._n_states <= 0:
            raise ValueError("Number of states must be positive.")

    def __call__(self, obs: np.ndarray) -> int:
        bin_indices = np.empty(self.num_dims, dtype=np.int_)
        for i in range(self.num_dims):
            clipped_obs_i = np.clip(obs[i], self.obs_low[i], self.obs_high[i])
            if self.bins_per_dim[i] == 1:
                bin_indices[i] = 0
            else:
                thresholds = self.bin_edges[i][1:-1]
                bin_indices[i] = np.digitize(clipped_obs_i, thresholds)
        
        state_index = 0
        for i in range(self.num_dims):
            state_index = state_index * self.bins_per_dim[i] + bin_indices[i]
        return int(state_index)

    @property
    def n_states(self) -> int:
        return self._n_states


def main(args: argparse.Namespace):
    np.random.seed(args.seed)

    env = LunarLanderPCH()

    temp_gym_env = gym.make("LunarLander-v3")
    obs_low = temp_gym_env.observation_space.low
    obs_high = temp_gym_env.observation_space.high
    temp_gym_env.close()

    if args.algo == "ucbvi":
        # lander_bins_config = [8, 8, 4, 4, 8, 4, 2, 2]  # Original: 131,072 states -> ~2TiB for P_ss'aa'
        lander_bins_config = [3, 3, 2, 2, 3, 2, 2, 2]  # Reduced for UCBVI: 3*3*2*2*3*2*2*2 = 864 states
    elif args.algo == "ucbq":
        lander_bins_config = [6, 6, 4, 4, 6, 4, 2, 2]  # Finer for UCBQ: 6*6*4*4*6*4*2*2 = 55,296 states
    else: # Should not happen due to argparse choices
        lander_bins_config = [3, 3, 2, 2, 3, 2, 2, 2] 

    discretiser = Discretiser(obs_low, obs_high, lander_bins_config)

    if discretiser.n_states >= 10**5: # Adjusted warning threshold
        print(f"Warning: Number of states ({discretiser.n_states}) is large. Training might be slow or memory intensive.")
    elif discretiser.n_states >= 10**6:
        print(f"Warning: Number of states ({discretiser.n_states}) is very large. Training might be slow.")

    if args.algo == "ucbvi":
        agent = UCBVI(
            num_states=discretiser.n_states,
            n_actions=env.action_space.n,
            horizon=args.horizon,
            delta=args.delta
        )
    elif args.algo == "ucbq":
        if UCBQ is None:
            print("Error: UCBQ algorithm selected, but ucbq.py could not be imported.")
            sys.exit(1)
        agent = UCBQ(
            n_states=discretiser.n_states, # UCBQ uses n_states
            n_actions=env.action_space.n,
            delta=args.delta
            # UCBQ does not take horizon or epsilon in its __init__
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    returns: List[float] = []
    start_time = time.time()

    print(f"Starting training with {args.algo.upper()} for {args.episodes} episodes...")
    print(f"Discretised states: {discretiser.n_states}, Actions: {env.action_space.n}")
    if args.verbose:
        print(f"Observation space low: {obs_low}")
        print(f"Observation space high: {obs_high}")
        print(f"Bins per dimension: {lander_bins_config}")


    for ep in range(args.episodes):
        obs_continuous, _ = env.reset(seed=args.seed + ep)
        s = discretiser(obs_continuous)
        ep_return = 0.0
        
        if args.verbose and ep < 2 : # Verbose for first 2 episodes
            print(f"\n--- Episode {ep+1} (Verbose) ---")
            print(f"  Initial raw obs: {np.round(obs_continuous, 2)}")
            print(f"  Initial discrete state s: {s}")

        for t in range(args.horizon):
            if args.algo == "ucbvi":
                # Step 1: Determine the best intended action x_int based on V values
                x_intended = np.argmax(agent.V[s, :])  # V has shape (S, A_intended)
                # Step 2: Get the actual action to apply to the environment based on s and x_intended
                a_to_apply = agent.act(s, x_intended) # agent.act(s, x_int) returns the action to take
            elif args.algo == "ucbq":
                a_to_apply = agent.act(s) # UCBQ.act(s)
            else: # Should not happen
                a_to_apply = env.action_space.sample()

            obs2_continuous, r, terminated, truncated, info = env.step(a_to_apply) # Correct unpacking
            s2 = discretiser(obs2_continuous)
            done = terminated or truncated # Define done for loop logic
            
            if args.algo == "ucbvi":
                agent.update(s, x_intended, a_to_apply, r, s2)
            elif args.algo == "ucbq":
                # realised_action = info.get("realised_action", a_to_apply) # UCBQ update doesn't use realised_action explicitly
                agent.update(s, a_to_apply, r, s2) # UCBQ.update(s, a, r, s_next)
            
            ep_return += r
            
            if args.verbose and ep < 2 and t < 5: # Verbose for first 5 steps of first 2 episodes
                print(f"  Step {t+1}:")
                if args.algo == "ucbvi":
                    print(f"    s: {s}, intended_x: {x_intended}, applied_a: {a_to_apply}, r: {r:.2f}")
                elif args.algo == "ucbq":
                    print(f"    s: {s}, applied_a: {a_to_apply}, r: {r:.2f}")
                print(f"    raw_obs_next: {np.round(obs2_continuous, 2)}")
                print(f"    s_next: {s2}, term: {terminated}, trunc: {truncated}")

            s = s2 # current_state = next_state

            # Original check for successful landing (restored)
            if terminated and r >= 100: 
                print(f"Ep {ep+1:4d}, Step {t+1}: SUCCESSFUL LANDING! Reward: {r:.2f}, Total Return: {ep_return:.2f}")
            
            if done: # If episode is done (either terminated or truncated)
                break
        
        returns.append(ep_return)
        
        if (ep + 1) % max(1, args.episodes // 10) == 0 or ep == args.episodes -1 :
             print(f"Ep {ep+1:4d}/{args.episodes} | Return {ep_return:7.2f} | Steps {t+1}")

    env.close()
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f} seconds.")

    # Calculate and print average returns for early vs. late episodes
    if args.episodes >= 20: # Ensure enough episodes for a 10% slice
        num_slice = args.episodes // 10
        avg_return_first_10_percent = np.mean(returns[:num_slice])
        avg_return_last_10_percent = np.mean(returns[-num_slice:])
        print(f"Average return of first {num_slice} episodes: {avg_return_first_10_percent:.2f}")
        print(f"Average return of last  {num_slice} episodes: {avg_return_last_10_percent:.2f}")
        if avg_return_last_10_percent > avg_return_first_10_percent:
            print(f"Performance improved by {avg_return_last_10_percent - avg_return_first_10_percent:.2f} on average.")
        else:
            print(f"Performance did not clearly improve (or worsened by {avg_return_first_10_percent - avg_return_last_10_percent:.2f}).")

    if args.plot:
        plot_filename = f"lander_{args.algo}_returns.png"
        print(f"Saving episode returns plot to {plot_filename}...")
        plt.figure(figsize=(10, 6))
        plt.plot(returns) # Plotting actual returns
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Episode Return")
        plt.title(f"Episode Returns for {args.algo.upper()} on LunarLander (Discretised)")
        plt.grid(True)
        plt.savefig(plot_filename)
        # if args.show_visuals: # Commenting out plt.show() for non-interactive saving
        #     print("Displaying regret plot...")
        #     plt.show()
        plt.close() # Ensure plot is closed to free memory

    if args.gif:
        print("Generating GIF frames of a trained policy rollout...")
        gif_env = LunarLanderPCH(render_mode="rgb_array")
        frames: List[np.ndarray] = []
        obs_gif, _ = gif_env.reset(seed=args.seed + args.episodes + 1)
        s_gif = discretiser(obs_gif)
        ep_return_gif = 0.0
        
        for step_gif in range(args.horizon):
            frame = gif_env.render()
            if frame is not None:
                 frames.append(frame)
            
            if args.algo == "ucbvi":
                x_intended_gif = np.argmax(agent.V[s_gif, :])
                a_to_apply_gif = agent.act(s_gif, x_intended_gif)
            elif args.algo == "ucbq":
                a_to_apply_gif = agent.act(s_gif) # UCBQ.act(s)
            else: # Should not happen
                a_to_apply_gif = gif_env.action_space.sample()
            
            obs2_gif, r_gif, terminated_gif, truncated_gif, _ = gif_env.step(a_to_apply_gif) # Unpack 5 values
            s2_gif = discretiser(obs2_gif)
            s_gif = s2_gif
            ep_return_gif += r_gif
            done_gif = terminated_gif or truncated_gif # Define done_gif

            if terminated_gif and r_gif >= 100: # Check for successful landing in GIF rollout
                print(f"GIF Rollout, Step {step_gif+1}: SUCCESSFUL LANDING! Reward: {r_gif:.2f}, Total Return: {ep_return_gif:.2f}")

            if done_gif:
                break
        
        gif_env.close()
        print(f"GIF rollout return: {ep_return_gif:.2f}, Steps: {step_gif+1}")

        if frames:
            if args.show_visuals and len(frames) > 0:
                print("Displaying sample frames from GIF rollout...")
                num_sample_frames = min(len(frames), 3)
                sample_indices = np.linspace(0, len(frames) - 1, num_sample_frames, dtype=int)
                
                fig_samples, axes_samples = plt.subplots(1, num_sample_frames, figsize=(5 * num_sample_frames, 5))
                if num_sample_frames == 1: axes_samples = [axes_samples] # Make iterable

                for i, frame_idx in enumerate(sample_indices):
                    axes_samples[i].imshow(frames[frame_idx])
                    axes_samples[i].set_title(f"Frame {frame_idx}")
                    axes_samples[i].axis('off')
                plt.suptitle("Sample GIF Frames")
                # plt.show() # Commenting out plt.show() for non-interactive saving
                # Save the sample frames plot instead of showing it, if desired, or just remove display
                sample_frames_filename = "lander_sample_frames.png"
                plt.savefig(sample_frames_filename)
                print(f"Saved sample GIF frames to {sample_frames_filename}")
                plt.close(fig_samples) # Ensure plot is closed

            print("Saving lander_rollout.gif...")
            fig_gif_anim = plt.figure(figsize=(frames[0].shape[1]/100, frames[0].shape[0]/100), dpi=100) # Adjust size
            plt.axis('off')
            patch_anim = plt.imshow(frames[0])
            
            def animate_fn(i):
                patch_anim.set_data(frames[i])
                return patch_anim,

            anim = animation.FuncAnimation(fig_gif_anim, animate_fn, frames=len(frames), interval=50, blit=True)
            try:
                anim.save("lander_rollout.gif", writer="pillow", fps=20)
                print("Saved lander_rollout.gif")
            except Exception as e_gif:
                print(f"Error saving GIF: {e_gif}. Is Pillow installed? (`pip install Pillow`)")
            plt.close(fig_gif_anim)
        else:
            print("No frames collected for GIF.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UCB-VI or UCB-Q on LunarLanderPCH with discretised states.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--horizon", type=int, default=1000, help="Maximum steps per episode.")
    parser.add_argument("--delta", type=float, default=0.1, help="Delta parameter for UCB algorithms.")
    parser.add_argument("--algo", type=str, default="ucbvi", choices=["ucbvi", "ucbq"], help="Algorithm to use.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--gif", action="store_true", help="Save a GIF of a trained policy rollout.")
    parser.add_argument("--plot", action="store_true", help="Save a PNG plot of the regret curve.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose printouts for debugging.")
    parser.add_argument("--show-visuals", action="store_true", help="Show plots/GIF frames interactively instead of just saving.")
    
    print("Running Discretiser self-test...")
    _test_gym_env = gym.make("LunarLander-v3")
    _test_obs_space = _test_gym_env.observation_space
    _d_test = Discretiser(_test_obs_space.low, _test_obs_space.high, [8, 8, 4, 4, 8, 4, 2, 2])
    _test_zeros_obs = np.zeros(8)
    _test_idx = _d_test(_test_zeros_obs)
    assert 0 <= _test_idx < _d_test.n_states, f"Test failed: idx={_test_idx}, n_states={_d_test.n_states}"
    assert _d_test.n_states < 10**6, f"Test failed: n_states={_d_test.n_states} >= 10^6"
    _test_gym_env.close()
    print("Discretiser self-test passed.")
    
    parsed_args = parser.parse_args()
    main(parsed_args) 