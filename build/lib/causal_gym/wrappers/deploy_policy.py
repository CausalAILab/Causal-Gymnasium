import gymnasium as gym

class DeployPolicy(gym.Wrapper):
    def __init__(self, env, behavior_policy):
        super().__init__(env)
        self.behavior_policy = behavior_policy

    def step(self, state):
        action = self.behavior_policy[state]
        next_state, reward, done, terminated, info = self.env.step(action)
        return action, next_state, reward, done, terminated, info
