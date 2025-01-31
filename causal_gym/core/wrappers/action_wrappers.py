from __future__ import annotations
from enum import IntEnum
from typing import Any, SupportsFloat
from gymnasium import spaces
from causal_gym.core import ActionPCHWrapper
from causal_gym.envs import WindyMiniGridPCH
from gymnasium.core import WrapperActType, ActType, ObsType, Env, Wrapper


class Actions(IntEnum):
    up = 0
    down = 1
    left = 2
    right = 3
    still = 4

class MiniGridActionRemapWrapper(ActionPCHWrapper):

    def __init__(self, env: WindyMiniGridPCH):
        """Constructor for the action wrapper.

        Args:
            env: Environment to be wrapped.
        """
        assert isinstance(env, WindyMiniGridPCH),f"This only works with WindyMiniGridPCH, not '{type(env)}'"
        ActionPCHWrapper.__init__(self, env)
        self.env = env
        self.actions = Actions
        self.action_space = spaces.Discrete(len(self.actions))

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            return self.env.__getattr__(name)
        
    @property
    def agent_dir(self,):
        return self.env.agent_dir
    
    @agent_dir.setter
    def agent_dir(self, new_dir):
        self.env.agent_dir = new_dir

    @property
    def agent_pos(self,):
        return self.env.agent_pos
    
    @agent_pos.setter
    def agent_pos(self, new_pos):
        self.env.agent_pos = new_pos

    @property
    def wind_dir(self,):
        return self.env.wind_dir
    
    @wind_dir.setter
    def wind_dir(self, new_dir):
        self.env.wind_dir = new_dir

    def see(self, bpolicy=None) -> tuple[WrapperActType, ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`see` action using :meth:`self.wrap_action`."""
        if bpolicy is not None:
            unwrapped_bpolicy = lambda state, wind: self.unwrap_action(bpolicy(state, wind))
        else:
            unwrapped_bpolicy = None
        prev_dir = self.env.agent_dir
        action, observation, reward, terminated, truncated, info = self.env.see(bpolicy=unwrapped_bpolicy)
        new_dir = self.env.agent_dir
        return self.wrap_action(action, prev_dir, new_dir), observation, reward, terminated, truncated, info

    def do(self, action: WrapperActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runs the :attr:`env` :meth:`env.do` using the modified ``action`` from :meth:`self.unwrap_action`."""
        return self.env.do(self.unwrap_action(action))

    def wrap_action(self, action: ActType, prev_dir: int, new_dir: int) -> WrapperActType:
        """
        Map the actions from the MiniGrid system of direction + three way moves to four-way moves

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        if action == 6:
            action = self.actions.still
        elif new_dir == 0:
            action = self.actions.right
        elif new_dir == 1:
            action = self.actions.down
        elif new_dir == 2:
            action = self.actions.left
        elif new_dir == 3:
            action = self.actions.up
        return action

    def unwrap_action(self, action: WrapperActType) -> ActType:
        """
        Map the actions from four-way moves to the MiniGrid system of direction + three way moves

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        if action == self.actions.left:
            self.env.agent_dir = 2
            action = 2
        elif action == self.actions.right:
            self.env.agent_dir = 0
            action = 2
        elif action == self.actions.up:
            self.env.agent_dir = 3
            action = 2
        elif action == self.actions.down:
            self.env.agent_dir = 1
            action = 2
        elif action == self.actions.still:
            action = 6
        return action
