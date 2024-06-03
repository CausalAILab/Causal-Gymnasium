import gymnasium as gym
from gymnasium import spaces
from causal_gym import SCM
import pygame
import numpy as np


class GridWorldEnv(SCM):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),  #right
            1: np.array([0, 1]),  #down
            2: np.array([-1, 0]), #left
            3: np.array([0, -1]), #up
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
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def action(self):
        return self.action_space.sample()
    
    def observation(self):
        return self._get_obs()
    
    def see(self):
        return self.step(self.action())
    
    def do(self, action):
        return self.step(action)
    
    def reset(self, seed=None, options=None, agent_location=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the target's location uniformly at center
        self._target_location = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )

        if agent_location is not None:
            self._agent_location = agent_location
        else:
            self._agent_location = self._target_location
            while np.array_equal(self._agent_location, self._target_location):
                self._agent_location = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        #if self._wind_direction == 0:
        # We use `np.clip` to make sure we don't leave the grid:
        #    self._agent_location = np.clip(
        #        self._agent_location + direction, 0, self.size - 1
        #    )
        #elif action != 0:
            # We use `np.clip` to make sure we don't leave the grid:
        #    self._agent_location = np.clip(
        #        self._agent_location + wind, 0, self.size - 1
        #    )

        #if action != 0:
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An episode is done iff the agent has reached the target
        done = np.array_equal(self._agent_location, self._target_location)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, -1, done, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        offset = 50

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size + 2*offset, self.window_size + 2*offset))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size + 2*offset, self.window_size + 2*offset))
        canvas.fill((0, 0, 0))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (50, 205, 50),
            pygame.Rect(
                self._target_location * pix_square_size + offset,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (255, 69, 0),
            (self._agent_location + 0.5) * pix_square_size + offset,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(1, self.size + 2):
            pygame.draw.line(
                canvas,
                (112, 112, 112),
                (offset, pix_square_size * x + offset),
                (self.window_size + offset, pix_square_size * x + offset),
                width=5,
            )
            pygame.draw.line(
                canvas,
                (112, 112, 112),
                (pix_square_size * x + offset, offset),
                (pix_square_size * x + offset, self.window_size + offset),
                width=5,
            )

        pygame.draw.rect(
            canvas,
            (112, 112, 112),
            pygame.Rect(
                np.array([0, 0], dtype=int),
                (self.window_size + 2*offset, offset),
            ),
        )
        
        pygame.draw.rect(
            canvas,
            (112, 112, 112),
            pygame.Rect(
                np.array([0, 0], dtype=int),
                (offset, self.window_size + 2*offset),
            ),
        )

        pygame.draw.rect(
            canvas,
            (112, 112, 112),
            pygame.Rect(
                np.array([self.window_size + offset, 0], dtype=int),
                (offset, self.window_size + 2*offset),
            ),
        )

        pygame.draw.rect(
            canvas,
            (112, 112, 112),
            pygame.Rect(
                np.array([0, self.window_size + offset], dtype=int),
                (self.window_size + 2*offset, offset),
            ),
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

