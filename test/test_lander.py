# %%
"""
test_lander.py
--------------
Observational / interventional baselines and Counterfactual UCB-VI
experiment on LunarLander-v2 through the causal_gym wrapper.

Run:
    python test_lander.py
Generates two PNGs:
    • lander_probs_ctf_ucb.png
    • lander_ctf_ucb_regret.png
"""
import numpy as np
import matplotlib.pyplot as plt
from causal_gym.envs import LunarLanderPCH


NUM_STATES = 5 * 6 * 5 * 6 #900
N_EPISODES   = 250
HORIZON      = 400
PLAN_SWEEPS  = 200          # new
PLAN_PERIOD  = 3            # new
TARGET_ACTION = 0  # NOOP

# %%
# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def run_observational(env, n_episodes, horizon):
    success_count = 0
    print("Running Observational...") # Add start message
    for ep in range(n_episodes):
        if (ep + 1) % 100 == 0:
            print(f"  Obs episode {ep + 1}/{n_episodes}", flush=True) # Periodic update
        env.reset()
        success = False
        for _ in range(horizon):
            _, obs, r, done, _ = env.see()
            if done:
                success = (r == 100)  # built-in landing bonus
                break
        if success:
            success_count += 1
    print("Observational finished.") # Add end message
    return success_count # Return total count

def run_interventional(env, n_episodes, horizon, action):
    success_count = 0
    print("Running Interventional (do=NOOP)...") # Add start message
    for ep in range(n_episodes):
        if (ep + 1) % 100 == 0:
            print(f"  Int episode {ep + 1}/{n_episodes}", flush=True) # Periodic update
        env.reset()
        success = False
        for _ in range(horizon):
            obs, r, done, _ = env.do(action)
            if done:
                success = (r == 100)
                break
        if success:
            success_count += 1
    print("Interventional finished.") # Add end message
    return success_count # Return total count

def run_ctf_ucbvi(env, agent, n_episodes, horizon, best_p):
    cum_regret = np.zeros(n_episodes, dtype=float)
    successes  = []
    print("Running Ctf-UCBVI...") # Add start message
    for ep in range(n_episodes):
        if (ep + 1) % 100 == 0:
            current_avg_success = np.mean(successes) if successes else 0.0
            print(f"  UCB episode {ep + 1}/{n_episodes} | Avg Success: {current_avg_success:.3f} | Cum Regret: {cum_regret[ep-1] if ep > 0 else 0:.1f}", flush=True)
        env.reset()
        tot_reward = 0

        for _ in range(horizon):
            # 1) Observe under behavior policy
            x_int, obs, _, done_obs, _ = env.see()
            if done_obs:
                # episode ended under the behavior policy
                break

            # 2) Pick counterfactual-optimal action
            s   = agent.discretize(obs)
            act = agent.act(s, x_int)

            # 3) Execute the chosen action
            obs_n, r, done, _ = env.do(act)

            # 4) Log the transition
            s_n = agent.discretize(obs_n)
            agent.update(s, x_int, act, r, s_n)

            tot_reward += r
            if done:
                # episode ended after intervention
                break

        # 5) Re-plan after the episode
        # re-plan every PLAN_PERIOD episodes with a limited sweep count
        if (ep + 1) % PLAN_PERIOD == 0:
            agent.plan(num_sweeps=PLAN_SWEEPS)

        # 6) Compute success & regret
        landed = (tot_reward >= 100)
        successes.append(landed)
        regret = best_p - landed
        cum_regret[ep] = cum_regret[ep-1] + regret if ep > 0 else regret

    print("Ctf-UCBVI finished.") # Add end message
    # Calculate final average success rate here
    final_avg_success = np.mean(successes) if successes else 0.0
    return final_avg_success, cum_regret

# %%
# ------------------------------------------------------------
# Main experiment (compute probabilities & regret)
# ------------------------------------------------------------
env_obs = LunarLanderPCH()
obs_success_count = run_observational(env_obs, N_EPISODES, HORIZON)
p_obs = obs_success_count / N_EPISODES
print(f"P(goal|see) ≈ {p_obs:.3f}")

env_int = LunarLanderPCH()
int_success_count = run_interventional(env_int, N_EPISODES, HORIZON, TARGET_ACTION)
p_int = int_success_count / N_EPISODES
print(f"P(goal|do=NOOP) ≈ {p_int:.3f}")


# Calculate final regrets
final_obs_reg = N_EPISODES - obs_success_count
final_int_reg = N_EPISODES - int_success_count

# %% 
# ------------------------------------------------------------
# Regret Summary (mimicking test_frozenlake.py output)
# ------------------------------------------------------------
print("\n=== Landing Success Rate ===")
print(f"Observational (see)      : {p_obs:.3f}")
print(f"Interventional (do=NOOP) : {p_int:.3f}")

# %% 
# ------------------------------------------------------------
# Plotting: success rates and cumulative regret
# ------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(["Obs (see)", "Int (do)"], [p_obs, p_int])
plt.ylim(0, 1)
plt.ylabel("Landing Success Rate")
plt.title("LunarLander: see vs do(NOOP)")
plt.tight_layout()
plt.savefig("lander_probs_basic.png")
plt.show()

