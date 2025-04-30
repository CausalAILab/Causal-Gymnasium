# Updated test_frozenlake.py to use counterfactual UCB-VI (Alg.26)
import numpy as np
import matplotlib.pyplot as plt
from causal_gym.envs import FrozenLakePCH

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


if __name__ == '__main__':
    # 1. Observational
    env_obs = FrozenLakePCH(is_slippery=True)
    p_obs = run_observational(env_obs, N_EPISODES, HORIZON)
    print(f"P(goal|see) ≈ {p_obs:.3f}")

    # 2. Interventional(always RIGHT)
    env_int = FrozenLakePCH(is_slippery=True)
    p_int = run_interventional(env_int, N_EPISODES, HORIZON, TARGET_ACTION)
    print(f"P(goal|do=RIGHT) ≈ {p_int:.3f}")

    # 4. Plot probabilities
    labels = ['Obs (see)', 'Int (do)']
    probs  = [p_obs, p_int]
    plt.figure(figsize=(6,4))
    plt.bar(labels, probs)
    plt.ylim(0,1)
    plt.ylabel('Reach Probability')
    plt.title('FrozenLake: see vs do(RIGHT)')
    plt.savefig('frozenlake_probs_basic.png')
    plt.close()
