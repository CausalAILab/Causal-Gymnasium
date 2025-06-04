# test_lander_ucbvi_learning_viz.py
# -------------------------------------
# This script tests the UCBVI (Upper Confidence Bound for Value Iteration) 
# reinforcement learning algorithm on the CausalGym LunarLanderPCH environment.
#
# Key Functionalities:
# 1. Learns a policy using the CtfUCBDriver (UCBVI variant).
# 2. Discretizes the continuous state space of LunarLander for the UCBVI algorithm.
# 3. Visualizes the learning process:
#    - Saves an animated GIF of the *first successful landing* episode.
#    - Generates and saves a plot of the final step reward achieved in each episode 
#      over the course of learning, along with a moving average.
# 4. Prints detailed step-by-step information (state, actions, rewards) for the 
#    episode where the first successful landing occurs.
# 5. Counts and reports the total number of successful landings.
#
# Outputs:
# - GIF: .important/lunar_lander/lander_first_successful_run.gif
# - Plot: .important/lunar_lander/lander_total_episodic_rewards.png
# - Console logs detailing learning progress and first success.
#
# Note: LunarLander is a more complex environment; learning can take significant time.
# -------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import imageio # For GIF generation
import os # For creating directories

from causal_gym.envs import LunarLanderPCH
from ucbvi import CtfUCBDriver as UCBVI # Corrected import

# Discretiser and configuration (from original test_lander_ucbvi.py)
def discretize(obs):
    # obs = [x, y, vx, vy, theta, omega, legL, legR]
    x, y, vx, vy, theta, omega, legL, legR = obs # Unpack all 8
    
    # Updated clipping ranges
    x_clipped = np.clip(x, -1.0, 1.0)
    y_clipped = np.clip(y, 0.0, 1.5)
    vx_clipped = np.clip(vx, -1.0, 1.0)
    vy_clipped = np.clip(vy, -1.5, 0.5) # Focused on descent phase
    theta_clipped = np.clip(theta, -0.5, 0.5) # Radians, approx +/- 28 degrees
    omega_clipped = np.clip(omega, -1.0, 1.0) # Radians/sec

    # Updated number of bins for each dimension
    NX, NY, NVX, NVY, NTHETA, NOMEGA = 3, 4, 3, 3, 4, 4

    # Create K-1 thresholds for K bins. np.digitize will return 0 to K-1.
    x_thresholds = np.linspace(-1.0, 1.0, NX + 1)[1:-1]
    y_thresholds = np.linspace(0.0, 1.5, NY + 1)[1:-1]
    vx_thresholds = np.linspace(-1.0, 1.0, NVX + 1)[1:-1]
    vy_thresholds = np.linspace(-1.5, 0.5, NVY + 1)[1:-1]
    theta_thresholds = np.linspace(-0.5, 0.5, NTHETA + 1)[1:-1]
    omega_thresholds = np.linspace(-1.0, 1.0, NOMEGA + 1)[1:-1]

    x_bin  = int(np.digitize(x_clipped, x_thresholds))
    y_bin  = int(np.digitize(y_clipped, y_thresholds))
    vx_bin = int(np.digitize(vx_clipped, vx_thresholds))
    vy_bin = int(np.digitize(vy_clipped, vy_thresholds))
    theta_bin = int(np.digitize(theta_clipped, theta_thresholds))
    omega_bin = int(np.digitize(omega_clipped, omega_thresholds))
    
    return (((((x_bin * NY + y_bin) * NVX + vx_bin) * NVY + vy_bin) * NTHETA + theta_bin) * NOMEGA + omega_bin)

NUM_STATES = 3 * 4 * 3 * 3 * 4 * 4 # 1728
N_EPISODES   = 1000
MAX_EPISODE_STEPS = 400 # Renamed from HORIZON for clarity

# New UCBVI parameters based on tuning insights
PLANNING_SWEEPS = 100       # Number of planning iterations for UCBVI
ALGO_MAX_EP_REWARD = 250.0  # Estimated max episodic reward for Q-clipping and bonus scaling
C_BONUS = 1.0
DELTA = 0.1
VERBOSE_ALGO = False        # Set to True for detailed UCBVI logs

PLAN_PERIOD  = 3   # From original lander script
LANDING_SUCCESS_REWARD = 100.0

OUTPUT_DIR = ".important/lunar_lander"
GIF_FILENAME = os.path.join(OUTPUT_DIR, "lander_first_successful_run.gif")
REWARD_PLOT_FILENAME = os.path.join(OUTPUT_DIR, "lander_total_episodic_rewards.png")

# UCBVI Helper function for learning visualization
def run_ucbvi_learning_viz(env, agent, n_episodes, max_episode_steps_param):
    episode_total_rewards = []
    first_success_frames = []
    first_success_achieved = False
    successful_episodes_count = 0
    first_success_details_printed = False

    for ep in range(n_episodes):
        if (ep + 1) % PLAN_PERIOD == 0 or ep == 0: 
            print(f"Planning at episode {ep + 1}...")
            agent.plan()

        obs_tuple = env.reset()
        obs = obs_tuple[0]
        
        current_render_mode = env.env.render_mode # Access render_mode from SCM
        
        if not first_success_achieved and not first_success_details_printed:
            print(f"\n--- Episode {ep + 1}/{n_episodes} (Attempting first success) ---")

        current_episode_frames = []
        last_step_reward = 0.0 # Initialize with a float
        current_episode_cumulative_reward = 0.0 # Initialize for accumulating total reward

        for t in range(max_episode_steps_param): # Use the renamed parameter for episode length
            current_obs_for_print = obs # State at the start of this step t
            
            x_int, obs_t_see, r_see, term_see, trunc_see, see_info = env.see()
            
            if term_see or trunc_see:
                if not first_success_achieved and not first_success_details_printed:
                    print(f"  Step {t + 1}: Episode terminated/truncated during env.see(). Last obs: {obs_t_see[:4]}...")
                last_step_reward = r_see 
                current_episode_cumulative_reward += r_see # Accumulate reward
                
                # Capture frame if it's the first attempt and ended by see()
                if not first_success_achieved and current_render_mode == "rgb_array":
                    frame_for_gif = env.render(show_wind=True, show_natural_action=True)
                    if frame_for_gif is not None:
                        current_episode_frames.append(frame_for_gif)
                break 

            s = agent.discretize(obs_t_see)
            a = agent.act(s, x_int)
            
            obs_next, r, terminated, truncated, do_info = env.do(a)
            
            # Capture frame *after* env.do() to get the state resulting from action 'a'
            # This ensures the final frame of a successful episode is captured.
            if not first_success_achieved and current_render_mode == "rgb_array":
                frame_for_gif = env.render(show_wind=True, show_natural_action=True)
                if frame_for_gif is not None:
                    current_episode_frames.append(frame_for_gif)

            s_n = agent.discretize(obs_next)
            agent.observe(s, x_int, a, r, s_n) 
            last_step_reward = r
            current_episode_cumulative_reward += r # Accumulate reward
            
            if not first_success_achieved and not first_success_details_printed:
                print(f"  Step {t + 1}:")
                print(f"    State (before SCM action): {current_obs_for_print[:4]}...") 
                print(f"    SCM Intended Action (x_int from see()): {x_int}")
                print(f"    Agent Chosen Action (a): {a}")
                print(f"    Reward (r from do(a)): {r:.4f}")
                print(f"    Next State (obs_next from do(a)): {obs_next[:4]}...")
                print(f"    Terminated: {terminated}, Truncated: {truncated}")

            obs = obs_next

            if terminated or truncated:
                break
        
        episode_total_rewards.append(current_episode_cumulative_reward) # Store total reward

        if not first_success_achieved and round(last_step_reward) == LANDING_SUCCESS_REWARD and terminated:
            first_success_achieved = True
            # current_episode_frames already contains the last frame due to capture after do()
            first_success_frames = current_episode_frames 
            print(f"--- First success achieved at episode {ep + 1}! Details above. ---")
            first_success_details_printed = True
        elif not first_success_achieved and not first_success_details_printed and (ep + 1) < n_episodes:
             print(f"--- Episode {ep + 1} did not result in success (final reward: {last_step_reward:.2f}, total reward: {current_episode_cumulative_reward:.2f}). ---")
        
        if round(last_step_reward) == LANDING_SUCCESS_REWARD and terminated:
            successful_episodes_count += 1

        if (ep + 1) % 25 == 0: # Print more frequently for longer episodes
            print(f"Episode {ep + 1}/{n_episodes} completed. Avg total reward (last 25): {np.mean(episode_total_rewards[-25:]):.2f}")

    return episode_total_rewards, first_success_frames, successful_episodes_count


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env_viz = LunarLanderPCH(render_mode="rgb_array")
    
    agent_viz = UCBVI(n_states=NUM_STATES, 
                        n_actions=env_viz.action_space.n, 
                        horizon=PLANNING_SWEEPS,          # Planning iterations
                        max_episode_reward=ALGO_MAX_EP_REWARD, # Q-clipping/bonus scaling
                        delta=DELTA, 
                        c_bonus=C_BONUS,
                        verbose=VERBOSE_ALGO)
    agent_viz.discretize = discretize

    print(f"Starting UCBVI learning for LunarLander: {N_EPISODES} episodes, PlanningSweeps={PLANNING_SWEEPS}, MaxEpReward={ALGO_MAX_EP_REWARD}, CBonus={C_BONUS}, Delta={DELTA}...")
    total_rewards_log, gif_frames, total_successes = run_ucbvi_learning_viz(env_viz, agent_viz, N_EPISODES, MAX_EPISODE_STEPS)

    if gif_frames:
        try:
            print(f"Saving GIF of first successful LunarLander run to {GIF_FILENAME}...")
            imageio.mimsave(GIF_FILENAME, gif_frames, fps=15) # Faster FPS for lander
            print(f"GIF saved.")
        except Exception as e:
            print(f"Error saving GIF: {e}")
            print("Please ensure imageio and ffmpeg are installed: pip install imageio imageio[ffmpeg]")
    else:
        print("No successful LunarLander run recorded for GIF generation, or environment not in rgb_array mode for capture.")

    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards_log, label='Total Episodic Reward')
    if len(total_rewards_log) >= 25: # Moving average window
        moving_avg = np.convolve(total_rewards_log, np.ones(25)/25, mode='valid')
        plt.plot(np.arange(12, len(total_rewards_log) - 12), moving_avg, label='25-episode Moving Average', color='red', linestyle='--')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'LunarLander UCBVI: Total Episodic Reward (PlanS={PLANNING_SWEEPS}, MaxR={ALGO_MAX_EP_REWARD}, CB={C_BONUS})')
    plt.legend()
    plt.grid(True)
    plt.savefig(REWARD_PLOT_FILENAME)
    print(f"Reward plot saved to {REWARD_PLOT_FILENAME}")
    plt.close()

    print(f"Total successful LunarLander landings: {total_successes} out of {N_EPISODES}")

    env_viz.close()
    print("LunarLander visualization script finished.") 