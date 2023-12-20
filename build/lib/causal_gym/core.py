"""Core API for SCM Environment, SCMWrapper, ActionSCMWrapper, RewardSCMWrapper and ObservationSCMWrapper."""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

import numpy as np

from gymnasium import (
    Env, 
    Wrapper,
)

from gymnasium import spaces
from gymnasium.utils import RecordConstructorArgs, seeding


if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec, WrapperSpec

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")
PolicyType = TypeVar("PolicyType")


class SCM(
    Env[ObsType, ActType],
    Generic[PolicyType, ObsType, ActType],
):
    r"""The main Causal-Gym class for implementing SCM environments.

    The class encapsulates an environment with arbitrary causal mechanisms and interaction regimes through the :meth:`see` and :meth:`do` functions.
    An environment can be partially or fully observed by single agents.

    The main API methods that users of this class need to know are:

    - :meth:`action` - Sample an action from the behavior policy given the current state of the environment.
    - :meth:`observation` - Resturn the observed state of the environment at the current stage of action.
    - :meth:`see` - Updates an environment following the behavior policy returning the realized action, the next agent observation, the reward for taking that actions,
    - :meth:`do` - Updates an environment with actions returning the next agent observation, the reward for taking that actions.

    Environments have additional attributes for users to understand the implementation

    - :attr:`policy` - The behavior policy already deployed in the environment.

    """

    policy: PolicyType

    def action(self) -> ActType:
        """Sample an action from the behavior policy given an observed state.
        
        Returns:
            
        """
        raise NotImplementedError
    
    def observation(self) -> ObsType:
        """Sample an action from the behavior policy given an observed state.
        
        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the current state of the environment.
        """
        raise NotImplementedError

    def see(self) -> tuple[ActType, ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics following the behavior policy.

        Returns:
            action (ActType): a realized action following the behavior policy.
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """
        raise NotImplementedError

    def do(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """  
        raise NotImplementedError

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")
WrapperPolicyType = TypeVar("WrapperPolicyType")


class SCMWrapper(
    SCM[PolicyType, ObsType, ActType], 
    Wrapper[WrapperObsType, WrapperActType, ObsType, ActType],
    Generic[WrapperPolicyType, WrapperObsType, WrapperActType, PolicyType, ObsType, ActType],
):
    """Wraps a :class:`causal_gym.SCM` to allow a modular transformation of the :meth:`see`, :meth:`do`, :meth:`action`, and :meth:`observation' methods.

    This class is the base class of all wrappers to change the behavior of the underlying SCM.
    SCMWrappers that inherit from this class can modify the :attr:`action_space`, :attr:`observation_space`,
    :attr:`reward_range`, :attr:`metadata` and :attr:`policy` attributes, without changing the underlying SCM's attributes.
    Moreover, the behavior of the :meth:`see`, :meth:`do`, :meth:`action`, and :meth:`observation' methods can be changed by these wrappers.

    Some attributes (:attr:`spec`, :attr:`render_mode`, :attr:`np_random`) will point back to the wrapper's environment
    (i.e. to the corresponding attributes of :attr:`env`).

    Note:
        If you inherit from :class:`SCMWrapper`, don't forget to call ``super().__init__(env)``
    """

    def __init__(self, env: SCM[PolicyType, ObsType, ActType]):
        """Wraps an environment to allow a modular transformation of the :meth:`see`, :meth:`do`, :meth:`action`, and :meth:`observation' methods.

        Args:
            env: The environment to wrap
        """
        self.env = env

        assert isinstance(env.unwrapped, SCM)

        self._policy: WrapperPolicyType | None = None

        Wrapper.__init__(self, env)

    def action(self) -> WrapperActType:
        return self.env.action()
    
    def observation(self) -> WrapperObsType:
        return self.env.observation()
    
    def see(self) -> tuple[WrapperActType, WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.see()
    
    def do(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.do(action)
    
    @property
    def policy(
        self,
    ) -> PolicyType | WrapperPolicyType:
        """Return the :attr:`Env` :attr:`policy` unless overwritten then the wrapper :attr:`policy` is used."""
        if self._policy is None:
            return self.env.policy
        return self._policy

    @policy.setter
    def policy(self, policy: WrapperPolicyType):
        self._policy = policy


class ObservationSCMWrapper(
    SCMWrapper[PolicyType, WrapperObsType, ActType, PolicyType, ObsType, ActType], 
):
    """Modify observations from :meth:`Env.see` and :meth:`Env.do` using :meth:`wrap_observation` function.

    If you would like to apply a function to only the observation before
    passing it to the learning code, you can simply inherit from :class:`ObservationSCMWrapper` and overwrite the method
    :meth:`wrap_observation` to implement that transformation. The transformation defined in that method must be
    reflected by the :attr:`env` observation space. Otherwise, you need to specify the new observation space of the
    wrapper by setting :attr:`self.observation_space` in the :meth:`__init__` method of your wrapper.
    """

    def __init__(self, env: SCM[PolicyType, ObsType, ActType]):
        """Constructor for the observation wrapper."""
        SCMWrapper.__init__(self, env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self.wrap_observation(obs), info

    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.wrap_observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.wrap_observation(observation), reward, terminated, truncated, info

    def see(self) -> tuple[ActType, WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`see` using :meth:`self.wrap_observation` on the returned observations."""
        action, observation, reward, terminated, truncated, info = self.env.see()
        return action, self.wrap_observation(observation), reward, terminated, truncated, info
    
    def do(self, action: ActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`do` using :meth:`self.wrap_observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.do(action)
        return self.wrap_observation(observation), reward, terminated, truncated, info
    
    def observation(self) -> WrapperObsType:
        """Modifies the :attr:`env` after calling :meth:`observation` using :meth:`self.wrap_observation` on the returned observations."""
        return self.wrap_observation(self.env.observation())
    
    def wrap_observation(self, observation: ObsType) -> WrapperObsType:
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        raise NotImplementedError

class RewardSCMWrapper(
    SCMWrapper[PolicyType, ObsType, ActType, PolicyType, ObsType, ActType], 
):
    """Superclass of wrappers that can modify the returning reward from one stage of interaction.

    If you would like to apply a function to the reward that is returned by the base environment before
    passing it to learning code, you can simply inherit from :class:`RewardSCMWrapper` and overwrite the method
    :meth:`wrap_reward` to implement that transformation.
    This transformation might change the :attr:`reward_range`; to specify the :attr:`reward_range` of your wrapper,
    you can simply define :attr:`self.reward_range` in :meth:`__init__`.
    """
        
    def __init__(self, env: SCM[PolicyType, ObsType, ActType]):
        """Constructor for the Reward wrapper."""
        SCMWrapper.__init__(self, env)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.wrap_reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.wrap_reward(reward), terminated, truncated, info

    def see(self) -> tuple[ActType, ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`see` reward using :meth:`self.wrap_reward`."""
        action, observation, reward, terminated, truncated, info = self.env.see()
        return action, observation, self.wrap_reward(reward), terminated, truncated, info
    
    def do(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`do` reward using :meth:`self.wrap_reward`."""
        observation, reward, terminated, truncated, info = self.env.do(action)
        return observation, self.wrap_reward(reward), terminated, truncated, info    
    
    def wrap_reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        raise NotImplementedError


class ActionSCMWrapper(
    SCMWrapper[PolicyType, ObsType, WrapperActType, PolicyType, ObsType, ActType], 
):
    """Superclass of wrappers that can modify the action before :meth:`env.do` and returned from :meth:`env.see`.

    If you would like to apply a function to the action before passing it to the base environment,
    you can simply inherit from :class:`ActionSCMWrapper` and overwrite the method  :meth:`wrap_action` and :meth:`unwrap_action` to implement
    that transformation. The transformation defined in that method must take values in the base environment’s
    action space. However, its domain might differ from the original action space.
    In that case, you need to specify the new action space of the wrapper by setting :attr:`self.action_space` in
    the :meth:`__init__` method of your wrapper.
    """

    def __init__(self, env: SCM[PolicyType, ObsType, ActType]):
        """Constructor for the action wrapper."""
        SCMWrapper.__init__(self, env)

    def step(
        self, action: WrapperActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runs the :attr:`env` :meth:`env.step` using the modified ``action`` from :meth:`self.unwrap_action`."""
        return self.env.step(self.unwrap_action(action))
    
    def step(self) -> tuple[WrapperActType, ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action, observation, reward, terminated, truncated, info = self.env.step()
        return self.wrap_action(action), observation, reward, terminated, truncated, info

    def see(self) -> tuple[WrapperActType, ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`see` action using :meth:`self.wrap_action`."""
        action, observation, reward, terminated, truncated, info = self.env.see()
        return self.wrap_action(action), observation, reward, terminated, truncated, info
    
    def do(self, action: WrapperActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runs the :attr:`env` :meth:`env.do` using the modified ``action`` from :meth:`self.unwrap_action`."""
        return self.env.step(self.unwrap_action(action))
        
    def action(self) -> WrapperActType:
        """Modifies the :attr:`env` :meth:`action` using :meth:`self.wrap_action`."""
        return self.wrap_action(self.env.action())

    def wrap_action(self, action: ActType) -> WrapperActType:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        raise NotImplementedError
    
    def unwrap_action(self, action: WrapperActType) -> ActType:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        raise NotImplementedError
    
class PolicySCMWrapper(
    SCMWrapper[WrapperPolicyType, ObsType, ActType, PolicyType, ObsType, ActType], 
):
    """Superclass of wrappers that can modify the policy deployed in the environment.

    If you would like to deploy a policy to the base environment,
    you can simply inherit from :class:`PolicySCMWrapper` and overwrite the method  :meth:`action` and :meth:`see` to implement
    that transformation. The policy defined in that method must take values in the base environment’s
    action space. However, its domain might differ from the original action space.
    In that case, you need to specify the new action space of the wrapper by setting :attr:`self.action_space` in
    the :meth:`__init__` method of your wrapper.
    """

    def __init__(self, env: SCM[PolicyType, ObsType, ActType]):
        """Constructor for the action wrapper."""
        SCMWrapper.__init__(self, env)

    def action(self) -> ActType:
        """Modifies the :attr:`env` :meth:`action` using the new policy"""
        raise NotImplementedError
    
    def see(self) -> tuple[WrapperActType, ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runs the :attr:`env` :meth:`env.do` using the modified ``action`` from :meth:`self.action`."""
        raise NotImplementedError