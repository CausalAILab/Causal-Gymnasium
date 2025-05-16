import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Explicitly set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.animation import ArtistAnimation # Added for animation

# Add the workspace root to Python's path to allow for package-like imports
# Old path addition might be problematic after moving the file.
# Assuming 'causal2' (workspace root) is the main project directory and on Python path, 
# or that causalgym is installed.
# If running this script directly and causalgym is not installed, 
# you might need to adjust sys.path to point to the 'causal2' directory.
# For example, if 'causal2' is the parent of 'causalgym':
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
# Or more robustly, run from the root of the project.
# For now, let's assume the environment handles imports correctly or the script is run from project root.
# The original line was: sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# This would add `causalgym/test/learning` to the path, which is unlikely to be what's needed 
# for `from causalgym.causal_gym.envs ...` unless causalgym is structured oddly.

# Attempting a more robust path addition for when script is in causalgym/test/learning
# This adds the 'causal2' directory to the path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..')) # Adjust if structure is different
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from causalgym.causal_gym.envs import FrozenLakePCH
from causalgym.causal_gym.algorithms.ucbvi import UCBVI

def run_ucbvi_experiment(num_episodes: int, env_config: dict, 
                         episode_horizon: int, planning_sweeps: int, 
                         delta: float = 0.1,
                         agent_epsilon: float = 0.1,
                         capture_replay: bool = False): # New arg
    """
    Runs a UCBVI experiment on the FrozenLakePCH environment using the provided UCBVI class.

    Args:
        num_episodes (int): Number of episodes to run.
        env_config (dict): Configuration for the FrozenLakePCH environment.
        episode_horizon (int): The max steps per episode (H for agent's episode interaction).
        planning_sweeps (int): Number of sweeps for UCBVI's plan() method.
        delta (float): Confidence parameter for UCBVI.
        agent_epsilon (float): Epsilon parameter for UCBVI.
        capture_replay (bool): Whether to capture frames for replaying the last successful episode.

    Returns:
        tuple[list, list, list]: 
            - List of cumulative rewards for each episode.
            - List of average rewards per step for each episode.
            - List of frames from the last successful episode (if capture_replay is True).
    """
    original_render_mode = env_config.get("render_mode")
    if capture_replay and original_render_mode != "rgb_array":
        print("Warning: For replay capture, overriding env_config render_mode to 'rgb_array'.")
        env_config_for_run = env_config.copy()
        env_config_for_run["render_mode"] = "rgb_array"
    else:
        env_config_for_run = env_config

    print(f"Initializing environment with config: {env_config_for_run}")
    env = FrozenLakePCH(**env_config_for_run)
    env_desc_for_plan = env.unwrapped.desc # Get the description for terminal state handling
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n # This is the number of *possible* actions agent can intend
    
    # UCBVI's horizon parameter is often tied to episode length for theoretical reasons
    # and as default for planning sweeps.
    print(f"Initializing UCBVI agent with: num_states={n_states}, n_actions={n_actions}, horizon={episode_horizon}, delta={delta}, epsilon={agent_epsilon}")
    agent = UCBVI(num_states=n_states, 
                  n_actions=n_actions, 
                  horizon=episode_horizon, # Agent's H parameter, often related to episode length
                  delta=delta,
                  epsilon=agent_epsilon,  # Use passed agent_epsilon
                  seed=env_config_for_run.get('seed', 0)) # Pass seed to agent
    
    all_episode_rewards = []
    all_episode_avg_step_rewards = [] # New list for average reward per step
    rewards_window = []
    successful_episodes = [] # List to store episode numbers where goal was reached
    last_successful_episode_frames = []

    # Determine print interval for UCBVI diagnostics
    # Aim for about 5-10 diagnostic print blocks during the run
    if num_episodes <= 20: # if very few episodes, print every time
        print_interval = 1
    elif num_episodes <= 100:
        print_interval = num_episodes // 5
    else:
        print_interval = num_episodes // 10
    print_interval = max(1, print_interval) # Ensure it's at least 1

    for episode_idx in range(num_episodes):
        current_state_initial, info = env.reset()
        current_state = current_state_initial
        episode_reward = 0.0
        num_steps_this_episode = 0
        current_episode_frames = [] # Store frames for current episode if capturing
        
        # Determine if UCBVI internal diagnostics should be printed for this episode
        print_ucbvi_diags = (episode_idx + 1) % print_interval == 0 or episode_idx == num_episodes - 1
        if episode_idx == 0: print_ucbvi_diags = True # Always print for first episode

        if print_ucbvi_diags:
            print(f"\n--- Episode {episode_idx + 1} (UCBVI Diags ON) ---")

        agent.plan(num_sweeps=planning_sweeps, print_diagnostics=print_ucbvi_diags, env_desc=env_desc_for_plan)

        for h_step in range(episode_horizon):
            num_steps_this_episode = h_step + 1 # Track number of steps
            current_s_for_print = current_state # Save state before action for printing
            
            if capture_replay:
                # The FrozenLakeSCM render method is called by step if render_mode is rgb_array
                # but we need the frame *before* the step for the current state.
                # The PCH wrapper's render() should give us the current view.
                frame = env.render() 
                if frame is not None:
                    current_episode_frames.append(frame)
            
            # Use random tie-breaking for intended_action selection
            v_values_for_intent = agent.V[current_state, :]
            intended_action = agent.random_argmax(v_values_for_intent) # Assumes agent has random_argmax
            
            applied_action = agent.act(current_state, intended_action)

            next_state, reward, terminated, truncated, info = env.do(applied_action)
            actual_executed_action = info.get("wind_overrode_action_to", applied_action)
            
            # Pass print_diagnostics flag, also force print if reward > 0 for UCBVI.update
            # UCBVI.update will internally decide to print if r>0 OR its print_diagnostics is True OR s=14/s_next=15
            agent.update(current_state, intended_action, actual_executed_action, reward, next_state, print_diagnostics=print_ucbvi_diags)
            
            episode_reward += reward
            
            if reward == 1.0: # Only print GOAL for actual goal reward
                if not successful_episodes or successful_episodes[-1] != episode_idx + 1:
                    successful_episodes.append(episode_idx + 1)
                print(f"Ep {episode_idx+1}, Step {h_step+1}: s={current_s_for_print}, x_int={intended_action}, a_app={applied_action}, a_exec={actual_executed_action} -> r={reward:.2f}, s'={next_state} *** GOAL! ***")
                if capture_replay:
                    # Capture the frame of the goal state as well
                    frame_after_step = env.render()
                    if frame_after_step is not None:
                        current_episode_frames.append(frame_after_step)
                    last_successful_episode_frames = list(current_episode_frames) # Save this successful episode's frames
            elif print_ucbvi_diags and (h_step == 0 or h_step == episode_horizon - 1 or current_s_for_print == 14 or next_state == 15) :
                # Print details for first/last steps, or steps involving state 14 or 15 (goal)
                print(f"Ep {episode_idx+1}, Step {h_step+1}: s={current_s_for_print}, x_int={intended_action}, a_app={applied_action}, a_exec={actual_executed_action} -> r={reward:.2f}, s'={next_state}")

            if current_s_for_print == 14 and applied_action == 2: # State 14 and action RIGHT
                print(f"  >>> S14, A_RIGHT -> R={reward:.2f}, S_NEXT={next_state} (Term: {terminated})")

            if terminated or truncated:
                if capture_replay and not (reward == 1.0): # if terminated but not goal, still capture last frame if needed.
                    # This case might be redundant if goal state is always terminal and already captured.
                    # Only add if not already added by goal condition.
                    if not current_episode_frames or (len(last_successful_episode_frames) == 0 or current_episode_frames[-1] is not last_successful_episode_frames[-1]):
                         # Avoid double-adding if goal was just reached and terminated
                        frame_after_step = env.render()
                        if frame_after_step is not None:
                            current_episode_frames.append(frame_after_step)
                break
            current_state = next_state
        
        all_episode_rewards.append(episode_reward)
        avg_step_reward_this_episode = episode_reward / num_steps_this_episode if num_steps_this_episode > 0 else 0
        all_episode_avg_step_rewards.append(avg_step_reward_this_episode)

        rewards_window.append(episode_reward)
        if len(rewards_window) > 100:
            rewards_window.pop(0)

        if (episode_idx + 1) % 10 == 0 or episode_idx == num_episodes -1:
            avg_reward_last_100 = np.mean(rewards_window) if rewards_window else 0.0
            print(f"Episode {episode_idx + 1}/{num_episodes} completed. Cumulative Reward: {episode_reward:.2f}. Avg Cumulative Reward (last {len(rewards_window)} episodes): {avg_reward_last_100:.2f}")
            if print_ucbvi_diags and episode_idx < num_episodes -1 : # Add a separator if diags were on and it's not the last summary
                print(f"---------------------------------------")

    if successful_episodes:
        print(f"\nAgent reached the goal (reward == 1.0) in the following {len(successful_episodes)} episodes:")
        # Print first 10 and last 10 successful episodes if many, or all if few
        if len(successful_episodes) <= 20:
            print(successful_episodes)
        else:
            print(f"{successful_episodes[:10]} ... {successful_episodes[-10:]}")
    else:
        print("\n *** Agent NEVER reached the goal (reward == 1.0) during the entire run. ***")
            
    env.close()
    return all_episode_rewards, all_episode_avg_step_rewards, last_successful_episode_frames # Return frames

def plot_results(all_episode_rewards: list, all_episode_avg_step_rewards: list, title: str = "UCBVI Learning Curve"):
    """Plots the cumulative rewards per episode, its moving average, and average reward per step."""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Cumulative Reward per Episode on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward per Episode", color=color)
    ax1.plot(all_episode_rewards, label="Cumulative Reward per Episode", alpha=0.7, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    if len(all_episode_rewards) >= 10:
        window_size = min(50, len(all_episode_rewards) // 3) 
        if window_size < 10: window_size = 10 
        
        moving_avg_cumulative = np.convolve(all_episode_rewards, np.ones(window_size)/window_size, mode='valid')
        moving_avg_x_cumulative = np.arange(window_size - 1, len(all_episode_rewards))
        ax1.plot(moving_avg_x_cumulative, moving_avg_cumulative, label=f"Moving Avg Cumulative Reward ({window_size} episodes)", color='darkblue', linestyle='--')

    # Create a secondary y-axis for Average Reward per Step
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel("Average Reward per Step", color=color)  # we already handled the x-label with ax1
    ax2.plot(all_episode_avg_step_rewards, label="Average Reward per Step", alpha=0.6, color=color, linestyle=':')
    ax2.tick_params(axis='y', labelcolor=color)
    
    if len(all_episode_avg_step_rewards) >= 10:
        window_size_avg_step = min(50, len(all_episode_avg_step_rewards) // 3)
        if window_size_avg_step < 10: window_size_avg_step = 10

        moving_avg_step = np.convolve(all_episode_avg_step_rewards, np.ones(window_size_avg_step)/window_size_avg_step, mode='valid')
        moving_avg_x_step = np.arange(window_size_avg_step - 1, len(all_episode_avg_step_rewards))
        ax2.plot(moving_avg_x_step, moving_avg_step, label=f"Moving Avg Step Reward ({window_size_avg_step} episodes)", color='darkgreen', linestyle='-.')


    fig.suptitle(title)
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    
    ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.subplots_adjust(top=0.93) # Adjust top to make space for suptitle

    plot_filename = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "") + ".png"
    # Ensure the plots directory exists
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    full_plot_path = os.path.join(plots_dir, plot_filename)
    
    plt.savefig(full_plot_path)
    print(f"Plot saved as {full_plot_path}")
    # plt.show()

def replay_episode(frames, fig_title="Episode Replay", interval=200): # Increased interval for slower replay
    if not frames:
        print("No successful episode frames to replay.")
        return
    
    # Check if running in a headless environment or if a display is available
    # This check might be too aggressive on Windows. Relying on plt.show() try-except is better.
    # if os.environ.get('DISPLAY', '') == '' and os.environ.get('WAYLAND_DISPLAY', '') == '':
    #     if not os.environ.get('MPLBACKEND', '').lower().startswith('agg'):
    #         print("Warning: No display detected. Skipping animation replay. Set MPLBACKEND=agg to suppress this warning if running headlessly.")
    #         return

    print(f"Preparing replay of {len(frames)} frames...")
    
    # Ensure plt is in non-interactive mode for generating animation, then show will block.
    # If script is run in an interactive IPython session, plt.show() might not block by default.
    # However, for a script, it usually does.
    current_interactive_status = plt.isinteractive()
    if current_interactive_status:
        plt.ioff()

    fig = plt.figure(figsize=(6,6)) # Adjust size as needed
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    
    ims = []
    for i, frame in enumerate(frames):
        if frame is None:
            print(f"Warning: Frame {i} is None, skipping.")
            continue
        # Ensure frame is a NumPy array and has the correct dtype (e.g., uint8) if not already.
        # Pygame surfaces from render('rgb_array') should be fine.
        if not isinstance(frame, np.ndarray):
            print(f"Warning: Frame {i} is not a numpy array, type: {type(frame)}. Skipping.")
            continue
        im = ax.imshow(frame, animated=True)
        ims.append([im])
    
    if not ims:
        print("No valid frames were collected for replay.")
        if current_interactive_status: # Restore previous interactive status
            plt.ion()
        plt.close(fig) # Close the figure if no animation will be shown
        return

    ani = ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=2000) # Add repeat_delay
    
    fig.suptitle(fig_title)
    
    try:
        print("Displaying animation. Close the animation window to continue...")
        plt.show(block=True) # block=True to make it modal
    except Exception as e:
        print(f"Error displaying animation: {e}")
        print("Ensure you have a display environment (e.g., X11) and a suitable matplotlib backend.")
    finally:
        if current_interactive_status: # Restore previous interactive status
            plt.ion()
        # plt.close(fig) # Let's not close it immediately, user might want to re-interact if backend allows

if __name__ == "__main__":
    NUM_EPISODES = 500 # Lowered for testing
    EPISODE_HORIZON = 25 
    PLANNING_SWEEPS = EPISODE_HORIZON 
    DELTA = 0.01 
    AGENT_EPSILON = 0.3 # Increased epsilon for more exploration in act()

    # Decide if replay should be captured. True for testing, might be False for long runs.
    CAPTURE_REPLAY_FLAG = True 

    ENV_CONFIG = {
        "map_name": "4x4",
        "is_slippery": True, # Enable slippery mode
        "wind_probabilities": (0.5, 0.125, 0.125, 0.125, 0.125), # Wind enabled with 50% no wind, 12.5% each direction
        "render_mode": "rgb_array" if CAPTURE_REPLAY_FLAG else None, # Set to rgb_array for replay capture
        "seed": 42 # Use a fixed seed for reproducibility
    }
    
    print("Starting UCBVI experiment with epsilon-greedy act() on Custom WINDY FrozenLake...")
    
    cumulative_rewards_history, avg_step_rewards_history, captured_frames = run_ucbvi_experiment(
        num_episodes=NUM_EPISODES,
        env_config=ENV_CONFIG,
        episode_horizon=EPISODE_HORIZON,
        planning_sweeps=PLANNING_SWEEPS,
        delta=DELTA,
        agent_epsilon=AGENT_EPSILON,
        capture_replay=CAPTURE_REPLAY_FLAG
    )
    
    plot_title = f"UCBVI on {'Slippery ' if ENV_CONFIG['is_slippery'] else ''}{'Windy ' if ENV_CONFIG['wind_probabilities'] != (1,0,0,0,0) else ''}FrozenLake {ENV_CONFIG['map_name']}"
    plot_results(cumulative_rewards_history, avg_step_rewards_history, title=plot_title)
    
    if captured_frames and CAPTURE_REPLAY_FLAG:
        watch = input("Watch replay of the last successful episode? (y/n): ").lower()
        if watch == 'y':
            replay_episode(captured_frames, fig_title="Replay: Last Successful Episode")
    elif CAPTURE_REPLAY_FLAG:
        print("No successful episodes were recorded for replay.")

    print("Experiment complete.")
    final_avg_window = min(100, NUM_EPISODES) 
    print(f"Final average cumulative reward over last {final_avg_window} episodes: {np.mean(cumulative_rewards_history[-final_avg_window:] if cumulative_rewards_history else [0]):.2f}") 
    print(f"Final average of 'average reward per step' over last {final_avg_window} episodes: {np.mean(avg_step_rewards_history[-final_avg_window:] if avg_step_rewards_history else [0]):.3f}") 

    if current_interactive_status: # Restore previous interactive status
        plt.ion()
    plt.close(fig) # Ensure figure is closed after showing or error