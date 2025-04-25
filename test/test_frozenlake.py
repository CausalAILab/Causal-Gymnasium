# Updated test_frozenlake.py to use counterfactual UCB-VI (Alg.26)
import numpy as np
import matplotlib.pyplot as plt
from causal_gym.envs import FrozenLakePCH
from causal_gym.algos.ucbvi import UCBVI

# Parameters
N_EPISODES = 1000
HORIZON       = 100  # max steps per episode
TARGET_ACTION = 2    # 'Right'

def run_observational(env, n_episodes, horizon):
    reached = []
    for _ in range(n_episodes):
        _, _ = env.reset()
        goal = False
        for _ in range(horizon):
            x_int, obs, r_obs, done, _ = env.see()
            if done:
                goal = (r_obs == 1.0)
                break
        reached.append(goal)
    return np.mean(reached)


def run_interventional(env, n_episodes, horizon, action):
    reached = []
    for _ in range(n_episodes):
        _, _ = env.reset()
        goal = False
        for _ in range(horizon):
            obs, r, done, _ = env.do(action)
            if done:
                goal = (r == 1.0)
                break
        reached.append(goal)
    return np.mean(reached)


def run_ctf_ucbvi(env, agent, n_episodes, horizon, best_p):
    cum_regret = np.zeros(n_episodes)
    successes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        for t in range(horizon):
            # observe intended policy action
            x_int, obs_t, _, done_obs, _ = env.see()
            s = agent.discretize(obs_t)
            # choose counterfactual-optimal action
            a = agent.act(s, x_int)
            # apply the chosen action interventionaly
            obs_next, r, done, _ = env.do(a)
            s_n = agent.discretize(obs_next)
            # record transition (s, intended, actual)
            agent.update(s, x_int, a, r, s_n)
            total_reward += r
            obs_t = obs_next
            if done:
                break
        # re-plan with optimistic updates
        agent.plan()
        succ = (total_reward > 0)
        successes.append(succ)
        regret = best_p - succ
        cum_regret[ep] = (cum_regret[ep-1] + regret) if ep>0 else regret

    return np.mean(successes), cum_regret


if __name__ == '__main__':
    # 1. Observational
    env_obs = FrozenLakePCH(is_slippery=True)
    p_obs = run_observational(env_obs, N_EPISODES, HORIZON)
    print(f"P(goal|see) ≈ {p_obs:.3f}")

    # 2. Interventional(always RIGHT)
    env_int = FrozenLakePCH(is_slippery=True)
    p_int = run_interventional(env_int, N_EPISODES, HORIZON, TARGET_ACTION)
    print(f"P(goal|do=RIGHT) ≈ {p_int:.3f}")

    # 3. Counterfactual UCB-VI
    env_ctf = FrozenLakePCH(is_slippery=True)
    agent = UCBVI(n_actions=4, horizon=HORIZON, delta=0.1, num_states=16)
    # override discretize to use raw state index
    agent.discretize = lambda obs: int(obs)
    p_ucb, cum_regret = run_ctf_ucbvi(env_ctf, agent, N_EPISODES, HORIZON, p_int)
    print(f"P_ctf_ucb(goal) ≈ {p_ucb:.3f}")

    # 4. Plot probabilities
    labels = ['Obs (see)', 'Int (do)', 'Ctf-UCB']
    probs  = [p_obs, p_int, p_ucb]
    plt.figure(figsize=(6,4))
    plt.bar(labels, probs)
    plt.ylim(0,1)
    plt.ylabel('Reach Probability')
    plt.title('FrozenLake: see vs do(RIGHT) vs Ctf-UCBVI')
    plt.savefig('frozenlake_probs_ctf_ucb.png')
    plt.close()

    # 5. Cumulative regret plot
    plt.figure(figsize=(6,4))
    plt.plot(cum_regret, label='Ctf-UCBVI Regret')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Regret')
    plt.title('FrozenLake: Cumulative Regret (Ctf-UCBVI)')
    plt.legend()
    plt.grid(True)
    plt.savefig('frozenlake_ctf_ucb_regret.png')
    plt.close()

    # 6. Summary
    final_reg = int(cum_regret[-1])
    print(f"Final cumulative regret (Ctf-UCBVI): {final_reg}")
