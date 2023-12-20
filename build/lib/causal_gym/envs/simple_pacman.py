import gymnasium as gym
from gym import spaces
import pygame
import numpy as np


class SimplePacmanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_hight = 256  # The size of the PyGame window
        self.window_width = 512

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Discrete(2),
                "ghost": spaces.Discrete(2),
                "color": spaces.Discrete(2),
            }
        )

        # We have 4 actions, corresponding to "left", "right"
        self.action_space = spaces.Discrete(2)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: -1,
            1: 1,
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "ghost": self._ghost_location, "color": self._ghost_color}

    def _get_info(self):
        return {"distance": abs(self._agent_location - self._ghost_location)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(2)
        self._ghost_location = 1 - self._agent_location

        self._ghost_color = self.np_random.integers(2)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self):
        return self.step(self.action_space.sample())
    
    def step(self, action):

        #There is 0.1 chance the robot does not comply
        #if np.random.rand() < 0.05:
        #    action = 1 - action

        # Map the action (element of {0,1}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, 0, 1)
        # An episode is done iff the agent has reached the target
        terminated = self._agent_location == self._ghost_location

        if terminated:
            if self._ghost_color == 0:
                reward = 1
            else:
                reward = -2
        else:
            reward = 0

        if not terminated:
            self._ghost_color = self.np_random.integers(2)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_hight))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_hight))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_hight
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255*self._ghost_color, 0, 255*(1-self._ghost_color)),
            pygame.Rect(
                self._ghost_location * pix_square_size, 0,
                pix_square_size, pix_square_size
            ),
        )

        pygame.draw.circle(
            canvas,
            (255, 255, 0),
            (self._agent_location * pix_square_size + 0.5 * pix_square_size, 0.5 * pix_square_size),
            pix_square_size / 3,
        )

        pygame.draw.line(
            canvas,
            0,
            (pix_square_size, 0),
            (pix_square_size, self.window_hight),
            width=3,
        )
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
