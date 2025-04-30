# test_lander_ucbvi.py
# Contains the Ctf-UCBVI testing code extracted from test_lander.py
import numpy as np
import matplotlib.pyplot as plt
from causal_gym.envs import LunarLanderPCH
from causal_gym.algos.ucbvi import UCBVI

# %%
# ------------------------------------------------------------
# Discretiser and configuration (Copied from original test_lander.py)
# ------------------------------------------------------------
def discretize(obs):
    # obs = [x, y, vx, vy, theta, omega, legL, legR]
    x, y, vx, vy, *_ = obs
    # Clip observations to the discretization range before digitizing
    x_clipped = np.clip(x, -1.2, 1.2)
    y_clipped = np.clip(y, 0.0, 1.4)
    vx_clipped = np.clip(vx, -1.0, 1.0)
    vy_clipped = np.clip(vy, -1.5, 1.0)

    x_bin  = int(np.digitize(x_clipped,  np.linspace(-1.2, 1.2, 5))) # Indices 0-4
    y_bin  = int(np.digitize(y_clipped,  np.linspace( 0.0, 1.4, 6))) # Indices 0-5
    vx_bin = int(np.digitize(vx_clipped, np.linspace(-1.0, 1.0, 5))) # Indices 0-4
    vy_bin = int(np.digitize(vy_clipped, np.linspace(-1.5, 1.0, 6))) # Indices 0-5
    return (((x_bin * 6 + y_bin) * 5 + vx_bin) * 6 + vy_bin)

NUM_STATES = 5 * 6 * 5 * 6 #900
N_EPISODES   = 250
HORIZON      = 400
PLAN_SWEEPS  = 200
PLAN_PERIOD  = 3
TARGET_ACTION = 0  # NOOP, used as baseline for regret

# %%
# ------------------------------------------------------------
# Helper functions (Copied from original test_lander.py)
# ------------------------------------------------------------
# Needed to run baseline for regret calculation
def run_interventional(env, n_episodes, horizon, action):
    success_count = 0
    print("Running Interventional (do=NOOP) baseline...") # Add start message
    for ep in range(n_episodes):
        if (ep + 1) % 100 == 0:
            print(f"  Baseline Int episode {ep + 1}/{n_episodes}", flush=True) # Periodic update
        env.reset()
        success = False
        for _ in range(horizon):
            obs, r, done, _ = env.do(action)
            if done:
                success = (r == 100)
                break
        if success:
            success_count += 1
    print("Interventional baseline finished.") # Add end message
    return success_count # Return total count

# UCBVI Helper function (Copied from original test_lander.py)
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
        regret = best_p - landed # Regret is vs P(goal|do=NOOP)
        cum_regret[ep] = cum_regret[ep-1] + regret if ep > 0 else regret

    print("Ctf-UCBVI finished.") # Add end message
    # Calculate final average success rate here
    final_avg_success = np.mean(successes) if successes else 0.0
    return final_avg_success, cum_regret

# %%
# ------------------------------------------------------------
# Main experiment (Copied from original test_lander.py)
# ------------------------------------------------------------
if __name__ == '__main__':
    # 1. Run baseline Interventional (do=NOOP)
    env_int = LunarLanderPCH()
    int_success_count = run_interventional(env_int, N_EPISODES, HORIZON, TARGET_ACTION)
    p_int = int_success_count / N_EPISODES
    print(f"Baseline P(goal|do=NOOP) ≈ {p_int:.3f}")

    # 2. Run Ctf-UCBVI
    env_ctf = LunarLanderPCH()
    agent   = UCBVI(num_states=NUM_STATES, n_actions=4, horizon=HORIZON, delta=0.05)
    agent.discretize = discretize
    p_ucb, cum_reg = run_ctf_ucbvi(env_ctf, agent, N_EPISODES, HORIZON, p_int) # Use p_int as baseline for regret
    print(f"P_ctf_ucb(goal) ≈ {p_ucb:.3f}")

    # 3. Calculate final regret
    final_ucb_reg = int(cum_reg[-1]) # Already calculated based on p_int baseline

    # %% 
    # ------------------------------------------------------------
    # Regret Summary (Copied from original test_lander.py)
    # ------------------------------------------------------------
    print("\n=== Final Cumulative Regret (vs do=NOOP) ===")
    print(f"Ctf-UCBVI                : {final_ucb_reg}")
    print("\n=== Avg Regret / Episode (vs do=NOOP) ===")
    print(f"Ctf-UCBVI                : {final_ucb_reg/N_EPISODES:.3f}")

    # %% 
    # ------------------------------------------------------------
    # Plotting (Copied from original test_lander.py)
    # ------------------------------------------------------------
    # Plot success rates (baseline vs UCBVI)
    plt.figure(figsize=(6, 4))
    plt.bar(["Int (do=NOOP)", "Ctf-UCBVI"], [p_int, p_ucb])
    plt.ylim(0, 1)
    plt.ylabel("Landing Success Rate")
    plt.title("LunarLander: do(NOOP) vs Ctf-UCBVI")
    plt.tight_layout()
    plt.savefig("lander_probs_ctf_ucb.png")
    plt.show()

    # Plot cumulative regret
    plt.figure(figsize=(6, 4))
    plt.plot(cum_reg, label="Ctf-UCBVI Regret (vs do=NOOP)")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Regret")
    plt.title("LunarLander: Cumulative Regret (Ctf-UCBVI)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lander_ctf_ucb_regret.png")
    plt.show() 