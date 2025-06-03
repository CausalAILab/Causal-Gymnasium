# test_frozenlake_ucbvi_learning_viz.py

# Tests UCBVI learning for FrozenLake environment
# Generates a GIF of the first successful run
# Saves a plot of the final step rewards
# Prints the total number of successful episodes

import numpy as np
import matplotlib.pyplot as plt
import imageio # For GIF generation
import os # For creating directories

from causal_gym.envs import FrozenLakePCH
from ucbvi import CtfUCBDriver as UCBVI

# Parameters
N_EPISODES = 2000 # Reduced for quicker test with new horizon
HORIZON_STEPS = 100 # Max steps per episode
ALGO_HORIZON = 1.0  # Algorithm's view of max episodic reward
OUTPUT_DIR = ".important/frozenlake"
GIF_FILENAME = os.path.join(OUTPUT_DIR, "frozenlake_nonslippery_first_successful_run.gif")
REWARD_PLOT_FILENAME = os.path.join(OUTPUT_DIR, "frozenlake_nonslippery_final_step_rewards.png")

ACTION_MAP = {0: "LEFT ", 1: "DOWN ", 2: "RIGHT", 3: "UP   "}

# UCBVI Helper function for learning visualization
def run_ucbvi_learning_viz(env, agent, n_episodes, horizon_steps_param):
    episode_final_rewards = []
    first_success_frames = []
    first_success_achieved = False
    successful_episodes_count = 0
    first_success_details_printed = False # Flag to ensure we only print details once
    ncol = env.env.ncol # Get ncol from the underlying SCM for row/col calculation

    for ep in range(n_episodes):
        agent.plan() # UCBVI plans at the start of each episode
        obs, info = env.reset()
        current_render_mode = env.render_mode
        if not first_success_achieved or ep == 0: 
            if current_render_mode != "rgb_array":
                pass 

        current_episode_frames = []
        last_step_reward = 0
        
        if not first_success_achieved and not first_success_details_printed:
            print(f"\n--- Episode {ep + 1}/{n_episodes} (Attempting first success) ---")

        for t in range(horizon_steps_param): # Use the renamed parameter for episode length
            current_obs_for_print = obs 
            row, col = current_obs_for_print // ncol, current_obs_for_print % ncol
            
            x_int, obs_t_see, r_see, term_see, trunc_see, see_info = env.see()
            # obs_t_see is the state after the SCM's random action (wind or slip if any)
            # For printout consistency, we use current_obs_for_print for "State before SCM action"
            
            if term_see or trunc_see: # Episode ended due to SCM's action in see()
                if not first_success_achieved and not first_success_details_printed:
                    print(f"  Step {t + 1}: Episode terminated/truncated during env.see().")
                    print(f"    State (after SCM action in see()): {obs_t_see} (Row: {obs_t_see // ncol}, Col: {obs_t_see % ncol})")
                    print(f"    SCM Intended Action (x_int from see()): {ACTION_MAP.get(x_int, str(x_int))}")
                    print(f"    Reward from see(): {r_see:.4f}")
                last_step_reward = r_see
                # Capture frame if it's the first attempt and ended by see()
                if not first_success_achieved and env.render_mode == "rgb_array":
                    frame_for_gif = env.render()
                    if frame_for_gif is not None:
                        current_episode_frames.append(frame_for_gif)
                break 

            s = agent.discretize(obs_t_see) # Discretize state after SCM's action from see()
            a = agent.act(s, x_int) # Agent chooses action based on state from see()
            
            # Agent takes action 'a'
            obs_next, r, terminated, truncated, do_info = env.do(a)
            
            # Capture frame *after* env.do() to get the state resulting from agent's action 'a'
            if not first_success_achieved and env.render_mode == "rgb_array":
                frame_for_gif = env.render()
                if frame_for_gif is not None:
                    current_episode_frames.append(frame_for_gif)

            s_n = agent.discretize(obs_next)
            agent.observe(s, x_int, a, r, s_n) 
            last_step_reward = r
            
            if not first_success_achieved and not first_success_details_printed:
                next_row, next_col = obs_next // ncol, obs_next % ncol
                print(f"  Step {t + 1}:")
                print(f"    State (before agent action, after SCM from see()): {obs_t_see} (Row: {obs_t_see // ncol}, Col: {obs_t_see % ncol})")
                print(f"    SCM Intended Action (x_int from see()): {ACTION_MAP.get(x_int, str(x_int))}")
                print(f"    Agent Chosen Action (a): {ACTION_MAP.get(a, str(a))}")
                print(f"    Reward (r from do(a)): {r:.4f}")
                print(f"    Next State (obs_next from do(a)): {obs_next} (Row: {next_row}, Col: {next_col})")
                print(f"    Terminated: {terminated}, Truncated: {truncated}")
                if do_info.get('wind_action_component') is not None:
                    print(f"      Wind component action: {ACTION_MAP.get(do_info['wind_action_component'], str(do_info['wind_action_component']))}, Intermediate state: {do_info['intermediate_state_after_wind']}")
                    print(f"      Agent component action (from intermediate): {ACTION_MAP.get(do_info['agent_action_component'], str(do_info['agent_action_component']))}")

            obs = obs_next

            if terminated or truncated:
                break
        
        episode_final_rewards.append(last_step_reward)

        if not first_success_achieved and last_step_reward == 1.0 and terminated:
            first_success_achieved = True
            first_success_frames = current_episode_frames
            print(f"--- First success achieved at episode {ep + 1}! Details above. ---")
            first_success_details_printed = True
        elif not first_success_achieved and not first_success_details_printed and (ep + 1) < n_episodes :
             print(f"--- Episode {ep + 1} did not result in success (Final reward: {last_step_reward:.2f}). ---")
        
        if last_step_reward == 1.0 and terminated:
            successful_episodes_count += 1

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{n_episodes} completed. Avg final step reward (last 50): {np.mean(episode_final_rewards[-50:]):.2f}")

    return episode_final_rewards, first_success_frames, successful_episodes_count


if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize environment with is_slippery=False and rgb_array for GIF generation
    env_viz = FrozenLakePCH(is_slippery=False, render_mode="rgb_array") 
    
    # For FrozenLake, num_states is typically nrow * ncol. Assuming 4x4 default map.
    # If using a different map, this needs to be adjusted.
    map_desc = env_viz.env.desc
    num_states = map_desc.shape[0] * map_desc.shape[1]

    # Instantiate CtfUCBDriver with separated horizon for planning and max_episode_reward
    agent_viz = UCBVI(n_states=num_states, 
                        n_actions=env_viz.action_space.n, 
                        horizon=HORIZON_STEPS,         # Pass HORIZON_STEPS (e.g., 100) as planning sweeps
                        max_episode_reward=ALGO_HORIZON, # Pass ALGO_HORIZON (e.g., 2.0) for reward scaling/clipping
                        delta=0.1, 
                        c_bonus=1.0, 
                        verbose=True)
    agent_viz.discretize = lambda obs: int(obs)

    print(f"Starting UCBVI learning for NON-SLIPPERY FrozenLake: {N_EPISODES} episodes, Planning_Sweeps(H)={HORIZON_STEPS}, Max_Ep_Reward={ALGO_HORIZON}...")
    final_rewards, gif_frames, total_successes = run_ucbvi_learning_viz(env_viz, agent_viz, N_EPISODES, HORIZON_STEPS)

    # Save GIF of the first successful run
    if gif_frames:
        try:
            print(f"Saving GIF of first successful run to {GIF_FILENAME}...")
            imageio.mimsave(GIF_FILENAME, gif_frames, fps=5)
            print(f"GIF saved.")
        except Exception as e:
            print(f"Error saving GIF: {e}")
            print("Please ensure imageio is installed: pip install imageio imageio[ffmpeg]")
    else:
        print("No successful run recorded for GIF generation, or environment not in rgb_array mode.")

    # Plot and save episode final step rewards
    plt.figure(figsize=(10, 5))
    plt.plot(final_rewards, label='Final Step Reward per Episode')
    if len(final_rewards) >= 50:
        moving_avg = np.convolve(final_rewards, np.ones(50)/50, mode='valid')
        # Adjust x-axis for moving average to align properly
        plt.plot(np.arange(len(final_rewards) - len(moving_avg), len(final_rewards) - len(moving_avg) + len(moving_avg)) - (50//2) + 1, moving_avg, label='50-episode Moving Average', color='red', linestyle='--')
    
    plt.xlabel('Episode')
    plt.ylabel('Final Step Reward')
    plt.title(f'FrozenLake (Non-Slippery) UCBVI: Final Step Reward (PlanningH={HORIZON_STEPS}, MaxR={ALGO_HORIZON})')
    plt.legend()
    plt.grid(True)
    plt.savefig(REWARD_PLOT_FILENAME)
    print(f"Reward plot saved to {REWARD_PLOT_FILENAME}")
    plt.close()

    print(f"Total successful episodes: {total_successes}")

    env_viz.close()
    print("Visualization script finished.") 