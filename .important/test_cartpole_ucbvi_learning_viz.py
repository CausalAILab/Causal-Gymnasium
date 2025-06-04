# .important/test_cartpole_ucbvi_learning_viz.py
# ---------------------------------------------
# This script tests the UCBVI (Upper Confidence Bound for Value Iteration) 
# reinforcement learning algorithm on the CausalGym CartPoleWindPCH environment.
#
# Key Functionalities:
# 1. Learns a policy using the CtfUCBDriver (UCBVI variant).
# 2. Discretizes the continuous state space of CartPoleWind for the UCBVI algorithm.
# 3. Visualizes the learning process:
#    - Saves an animated GIF of the *first episode that reaches MAX_EPISODE_STEPS*.
#    - Generates and saves a plot of the total episodic reward achieved 
#      over the course of learning, along with a moving average.
# 4. Prints detailed step-by-step information (state, actions, rewards) for the 
#    episode where MAX_EPISODE_STEPS is first reached.
# 5. Counts and reports the total number of episodes reaching MAX_EPISODE_STEPS.
#
# Outputs:
# - GIF: .important/cartpole_wind/cartpole_first_max_steps_run.gif
# - Plot: .important/cartpole_wind/cartpole_total_episodic_rewards.png
# - Console logs detailing learning progress and first "successful" run.
# ---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import imageio # For GIF generation
import os # For creating directories
import sys

# Ensure .important is in the path for ucbvi import if script is run from elsewhere
# or if .important is not inherently a package.
# This assumes the script is in .important, so relative import works.
# If run from workspace root, and .important is not a package, adjustments might be needed.
try:
    from ucbvi import CtfUCBDriver as UCBVI
except ImportError:
    # Fallback if .important is not directly in sys.path and not a package
    # Go up one level (from .important) to workspace root, then into .important
    # This is a bit fragile and depends on execution context.
    # A better way is to ensure .important is added to PYTHONPATH or structure as a package.
    # For now, assuming direct import or that this script is run such that .important is discoverable.
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add workspace root
    from important.ucbvi import CtfUCBDriver as UCBVI


from causal_gym.envs import CartPoleWindPCH

# --- Discretization Function ---
def discretize_cartpole(obs):
    # obs = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    cart_pos, cart_vel, pole_angle, pole_vel = obs

    # Define clipping ranges based on typical CartPole-v1 limits
    # Cart Position: episode terminates if abs(cart_position) > 2.4
    cart_pos_clipped = np.clip(cart_pos, -2.4, 2.4)
    # Cart Velocity: No explicit limit, but let's use a practical range
    cart_vel_clipped = np.clip(cart_vel, -2.0, 2.0)
    # Pole Angle: episode terminates if abs(pole_angle) > 0.2095 rad (~12 degrees)
    pole_angle_clipped = np.clip(pole_angle, -0.2095, 0.2095)
    # Pole Angular Velocity: No explicit limit, but let's use a practical range
    pole_vel_clipped = np.clip(pole_vel, -2.0, 2.0)

    # Define number of bins for each dimension
    N_CART_POS, N_CART_VEL, N_POLE_ANGLE, N_POLE_VEL = 5, 5, 7, 5

    # Create K-1 thresholds for K bins. np.digitize will return 0 to K-1.
    cart_pos_thresholds = np.linspace(-2.4, 2.4, N_CART_POS + 1)[1:-1]
    cart_vel_thresholds = np.linspace(-2.0, 2.0, N_CART_VEL + 1)[1:-1]
    pole_angle_thresholds = np.linspace(-0.2095, 0.2095, N_POLE_ANGLE + 1)[1:-1]
    pole_vel_thresholds = np.linspace(-2.0, 2.0, N_POLE_VEL + 1)[1:-1]

    cart_pos_bin   = int(np.digitize(cart_pos_clipped, cart_pos_thresholds))
    cart_vel_bin   = int(np.digitize(cart_vel_clipped, cart_vel_thresholds))
    pole_angle_bin = int(np.digitize(pole_angle_clipped, pole_angle_thresholds))
    pole_vel_bin   = int(np.digitize(pole_vel_clipped, pole_vel_thresholds))
    
    state_index = (((cart_pos_bin * N_CART_VEL + cart_vel_bin) * N_POLE_ANGLE + pole_angle_bin) * N_POLE_VEL + pole_vel_bin)
    return state_index

# --- Parameters ---
NUM_STATES = 5 * 5 * 7 * 5  # 875
N_ACTIONS = 2  # CartPole: 0 (push left), 1 (push right)
N_EPISODES = 2000
MAX_EPISODE_STEPS = 200 # Default for CartPoleWindSCM

PLANNING_SWEEPS = 100
ALGO_MAX_EP_REWARD = float(MAX_EPISODE_STEPS) # Max reward is 1 per step
C_BONUS = 1.0
DELTA = 0.1
VERBOSE_ALGO = False

PLAN_PERIOD = 3

OUTPUT_DIR = ".important/cartpole_wind"
GIF_FILENAME = os.path.join(OUTPUT_DIR, "cartpole_first_max_steps_run.gif")
REWARD_PLOT_FILENAME = os.path.join(OUTPUT_DIR, "cartpole_total_episodic_rewards.png")

# --- Learning Loop Function ---
def run_ucbvi_cartpole_viz(env, agent, n_episodes, max_episode_steps_param):
    episode_total_rewards = []
    first_max_steps_frames = []
    first_max_steps_achieved = False
    max_steps_episodes_count = 0
    first_max_steps_details_printed = False

    for ep in range(n_episodes):
        if (ep + 1) % PLAN_PERIOD == 0 or ep == 0:
            if VERBOSE_ALGO: print(f"Planning at episode {ep + 1}...")
            agent.plan()

        obs_tuple = env.reset()
        obs = obs_tuple[0]
        
        current_render_mode = env.env.render_mode
        
        if not first_max_steps_achieved and not first_max_steps_details_printed:
            print(f"\n--- Episode {ep + 1}/{n_episodes} (Attempting first max_steps run) ---")

        current_episode_frames = []
        current_episode_cumulative_reward = 0.0
        episode_ended_early = False

        for t in range(max_episode_steps_param):
            current_obs_for_print = obs
            
            x_int, obs_t_see, r_see, term_see, trunc_see, see_info = env.see()
            
            if term_see or trunc_see: # Episode ended by SCM's internal policy or natural termination
                if not first_max_steps_achieved and not first_max_steps_details_printed:
                    print(f"  Step {t + 1}: Episode terminated/truncated during env.see(). Obs: {obs_t_see[:4]}..., Reward: {r_see:.2f}")
                current_episode_cumulative_reward += r_see
                if current_render_mode == "rgb_array" and not first_max_steps_achieved:
                    frame_for_gif = env.render()
                    if frame_for_gif is not None:
                        current_episode_frames.append(frame_for_gif)
                episode_ended_early = True
                break 

            s = agent.discretize(obs_t_see)
            a = agent.act(s, x_int)
            
            obs_next, r, terminated, truncated, do_info = env.do(a)
            
            if current_render_mode == "rgb_array" and not first_max_steps_achieved:
                frame_for_gif = env.render()
                if frame_for_gif is not None:
                    current_episode_frames.append(frame_for_gif)

            s_n = agent.discretize(obs_next)
            agent.observe(s, x_int, a, r, s_n)
            current_episode_cumulative_reward += r
            
            if not first_max_steps_achieved and not first_max_steps_details_printed:
                print(f"  Step {t + 1}:")
                print(f"    State (before SCM action): {current_obs_for_print[:4]}...")
                print(f"    SCM Intended Action (x_int): {x_int}, Agent Chosen Action (a): {a}")
                print(f"    Reward (r from do(a)): {r:.4f}, Total: {current_episode_cumulative_reward:.2f}")
                print(f"    Next State (obs_next from do(a)): {obs_next[:4]}...")
                print(f"    Terminated: {terminated}, Truncated: {truncated}")

            obs = obs_next

            if terminated or truncated:
                episode_ended_early = True
                break
        
        episode_total_rewards.append(current_episode_cumulative_reward)

        # Check for "success" (reaching max steps)
        if not episode_ended_early and (t + 1) == max_episode_steps_param:
            max_steps_episodes_count += 1
            if not first_max_steps_achieved:
                first_max_steps_achieved = True
                first_max_steps_frames = current_episode_frames
                print(f"--- First max_steps run achieved at episode {ep + 1}! Details above. ---")
                first_max_steps_details_printed = True
        elif not first_max_steps_achieved and not first_max_steps_details_printed and (ep + 1) < n_episodes:
             print(f"--- Episode {ep + 1} did not reach max_steps (ended at step {t+1}, total reward: {current_episode_cumulative_reward:.2f}). ---")
        
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep + 1}/{n_episodes} completed. Avg total reward (last 100): {np.mean(episode_total_rewards[-100:]):.2f}")

    return episode_total_rewards, first_max_steps_frames, max_steps_episodes_count

# --- Main Execution ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env_viz = CartPoleWindPCH(render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
    
    agent_viz = UCBVI(n_states=NUM_STATES, 
                        n_actions=N_ACTIONS, 
                        horizon=PLANNING_SWEEPS,
                        max_episode_reward=ALGO_MAX_EP_REWARD,
                        delta=DELTA, 
                        c_bonus=C_BONUS,
                        verbose=VERBOSE_ALGO)
    agent_viz.discretize = discretize_cartpole

    print(f"Starting UCBVI learning for CartPoleWind: {N_EPISODES} episodes, MaxSteps={MAX_EPISODE_STEPS}")
    print(f"Params: PlanSweeps={PLANNING_SWEEPS}, AlgoMaxEpReward={ALGO_MAX_EP_REWARD}, CBonus={C_BONUS}")
    
    total_rewards_log, gif_frames, total_max_steps_runs = run_ucbvi_cartpole_viz(env_viz, agent_viz, N_EPISODES, MAX_EPISODE_STEPS)

    if gif_frames:
        try:
            print(f"Saving GIF of first CartPoleWind max_steps run to {GIF_FILENAME}...")
            imageio.mimsave(GIF_FILENAME, gif_frames, fps=30)
            print(f"GIF saved.")
        except Exception as e:
            print(f"Error saving GIF: {e}")
            print("Please ensure imageio and ffmpeg are installed: pip install imageio imageio[ffmpeg]")
    else:
        print("No CartPoleWind run reached max_steps for GIF generation, or env not in rgb_array mode.")

    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards_log, label='Total Episodic Reward')
    if len(total_rewards_log) >= 50: # Moving average window
        moving_avg = np.convolve(total_rewards_log, np.ones(50)/50, mode='valid')
        plt.plot(np.arange(24, len(total_rewards_log) - 25), moving_avg, label='50-episode Moving Average', color='red', linestyle='--')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'CartPoleWind UCBVI: Total Episodic Reward (MaxS={MAX_EPISODE_STEPS}, PlanS={PLANNING_SWEEPS}, MaxR={ALGO_MAX_EP_REWARD}, CB={C_BONUS})')
    plt.legend()
    plt.grid(True)
    plt.savefig(REWARD_PLOT_FILENAME)
    print(f"Reward plot saved to {REWARD_PLOT_FILENAME}")
    plt.show() # Display the plot
    plt.close()

    print(f"Total episodes reaching max_steps ({MAX_EPISODE_STEPS}): {total_max_steps_runs} out of {N_EPISODES}")

    env_viz.close()
    print("CartPoleWind UCBVI visualization script finished.") 