# test_frozenlake_ucbvi.py
# Contains the UCBVI testing code extracted from test_frozenlake.py
import numpy as np
import matplotlib.pyplot as plt
from causal_gym.envs import FrozenLakePCH
from ucbvi import CtfUCBDriver as UCBVI

# Parameters (Copied from original test_frozenlake.py)
N_EPISODES = 1000
HORIZON       = 100  # max steps per episode
TARGET_ACTION = 2    # 'Right', used as baseline for regret

# Helper function (Copied from original test_frozenlake.py)
# Needed to run baseline for regret calculation
def run_interventional(env, n_episodes, horizon, action):
    reached = []
    for _ in range(n_episodes):
        _, _ = env.reset()
        goal = False
        for _ in range(horizon):
            obs, r, terminated, truncated, info = env.do(action)
            if terminated or truncated:
                goal = (r == 1.0 and terminated)
                break
        reached.append(goal)
    return np.mean(reached)

# UCBVI Helper function (Copied from original test_frozenlake.py)
def run_ctf_ucbvi(env, agent, n_episodes, horizon, best_p):
    cum_regret = np.zeros(n_episodes)
    successes = []

    for ep in range(n_episodes):
        agent.plan()
        obs, _ = env.reset()
        total_reward = 0
        for t in range(horizon):
            # observe intended policy action
            x_int, obs_t, _, _, _, _ = env.see()
            s = agent.discretize(obs_t)
            # choose counterfactual-optimal action
            a = agent.act(s, x_int)
            # apply the chosen action interventionaly
            obs_next, r, terminated, truncated, info = env.do(a)
            s_n = agent.discretize(obs_next)
            # record transition (s, intended, actual)
            agent.observe(s, x_int, a, r, s_n)
            total_reward += r
            obs_t = obs_next
            if terminated or truncated:
                break
        succ = (total_reward > 0)
        successes.append(succ)
        regret = best_p - succ
        cum_regret[ep] = (cum_regret[ep-1] + regret) if ep>0 else regret

    return np.mean(successes), cum_regret


if __name__ == '__main__':
    # 1. Run baseline interventional (always RIGHT) to get best_p for regret
    env_int = FrozenLakePCH(is_slippery=True)
    p_int = run_interventional(env_int, N_EPISODES, HORIZON, TARGET_ACTION)
    print(f"Baseline P(goal|do=RIGHT) ≈ {p_int:.3f}")

    # 2. Run Counterfactual UCB-VI
    env_ctf = FrozenLakePCH(is_slippery=True)
    agent = UCBVI(n_states=16, n_actions=4, horizon=HORIZON, delta=0.1)
    # override discretize to use raw state index
    agent.discretize = lambda obs: int(obs)
    p_ucb, cum_regret = run_ctf_ucbvi(env_ctf, agent, N_EPISODES, HORIZON, p_int)
    print(f"P_ctf_ucb(goal) ≈ {p_ucb:.3f}")

    # 3. Cumulative regret plot
    plt.figure(figsize=(6,4))
    plt.plot(cum_regret, label='Ctf-UCBVI Regret')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Regret')
    plt.title('FrozenLake: Cumulative Regret (Ctf-UCBVI)')
    plt.legend()
    plt.grid(True)
    plt.savefig('frozenlake_ctf_ucb_regret.png')
    plt.close()

    # 4. Summary
    final_reg = int(cum_regret[-1])
    print(f"Final cumulative regret (Ctf-UCBVI): {final_reg}") 