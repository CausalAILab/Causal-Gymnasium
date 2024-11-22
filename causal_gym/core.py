"""Core API for SCM Environment, PCHWrapper, ActionPCHWrapper, RewardPCHWrapper and ObservationPCHWrapper."""
from __future__ import annotations

import numpy as np
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Union

from collections import namedtuple
from gymnasium import Env, Wrapper
from gymnasium import spaces
from gymnasium.utils import RecordConstructorArgs, seeding

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec, WrapperSpec

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")
PolicyType = TypeVar("PolicyType")

# Define the policy scope as a named tuple
class PolicyScope(namedtuple('PolicyScope', ['act', 'obs'])):
    __slots__ = ()  # Named tuples are immutable, so we use __slots__ to prevent creating a __dict__
    def __new__(cls, act:list[ActType]=[], obs:list[ObsType]=[]):
        return super(PolicyScope, cls).__new__(cls, act, obs)
    

# Define the task as a named tuple
class Task(namedtuple('Task', ['learning_regime', 'str_assump', 'policy_space', 'reward_func'])):
    __slots__ = ()  # Named tuples are immutable, so we use __slots__ to prevent creating a __dict__
    def __new__(cls, learning_regime:str='see', str_assump:str='dag', policy_space:PolicyScope=None, reward_func:str='discount'):
        return super(Task, cls).__new__(cls, learning_regime, str_assump, policy_space, reward_func)
    

class SCM(
    Env[ObsType, ActType],
    Generic[PolicyType, ObsType, ActType],
):
    r"""The main Causal-Gym class for implementing SCM environments.

    The class encapsulates an environment with arbitrary causal mechanisms.
    An environment can be partially or fully observed by single agents.

    The main API methods that users of this class need to know are (other than the standard gymnasium APIs):
    - :meth:`observation` = Return the rendered environment observations at the current stage (potentially using render() APIs).
    - :meth:`action` - Sample an action from the behavior policy given the current state of the environment.
    - :meth:`get_graph` - Generate the causal diagram of the underlying environment.

    Environments have additional attributes for users to understand the implementation

    - :attr:`policy` - The behavior policy already deployed in the environment.

    """
    # Set this in SOME subclasses
    metadata: dict[str, Any] = {"render_modes": []}
    # define render_mode if your environment supports rendering
    render_mode: str | None = None
    reward_range = (-float("inf"), float("inf"))
    spec: EnvSpec | None = None

    # Set these in ALL subclasses
    action_space: spaces.Space[ActType]
    observation_space: spaces.Space[ObsType]

    policy: PolicyType | None = None
    _env: Env | None = None
    _np_random: np.random.Generator | None = None

    def action(self) -> ActType:
        """Sample an action from the behavior policy given an observed state.
        
        Returns:
            
        """
        raise NotImplementedError
    
    def observation(self) -> ObsType:
        """Get the current observations.
        
        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the current state of the environment.
        """
        raise NotImplementedError
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.

        Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        .. versionchanged:: v0.25

            The ``return_info`` parameter was removed and now info is expected to be returned.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        return super().reset(seed=seed, options=options)

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, _ = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value
    
    @property
    def get_graph(self,) -> tuple[dict[int, str], list[list[int]], list[list[int]]]:
        """Return the causal diagram of the environment.
        Returns:
            Nodes: a dictionary mapping from node index ([0, N-1]) to each node's semantic meaning.
            base_graph: an extended adjacent matrix representation of the directed graphical structure.  
                G[i,j] = -1 i<-j
                G[i,j] = 0 i j
                G[i,j] = 1 i->j
            conf_graph: a matrix representing the existence of confounders between nodes.
                G[i, j] = 0 no confounder
                G[i, j] = 1 i<->j
        """
        raise NotImplementedError
    
    @property
    def unwrapped(self,) -> Union[Env[ObsType, ActType], Generic[PolicyType, ObsType, ActType]]:
        # This will return the underlying environment without wrappers
        if isinstance(self._env, Wrapper):
            return self._env.unwrapped
        else:
            return self
        

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")
WrapperPolicyType = TypeVar("WrapperPolicyType")

class PCH(    
    Wrapper[WrapperObsType, WrapperActType, ObsType, ActType],
    Generic[WrapperPolicyType, WrapperObsType, WrapperActType, PolicyType, ObsType, ActType],
):
    """
    The main class for interacting with SCMs.
    Currently we support L1 and L2 interactions, namely,
        - :meth:`see` - Updates an environment following the behavior policy returning the realized action, the next agent observation, the reward for taking that actions,
        - :meth:`do` - Updates an environment with actions returning the next agent observation, the reward for taking that actions.
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

    def do(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
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


class PCHWrapper(
    PCH[PolicyType, ObsType, ActType]
):
    """Wraps a :class:`causal_gym.PCH` to allow a modular transformation of the :meth:`see`, :meth:`do`, :meth:`action`, and :meth:`observation' methods.

    This class is the base class of all wrappers to change the behavior of the underlying SCM.
    PCHWrappers that inherit from this class can modify the :attr:`action_space`, :attr:`observation_space`,
    :attr:`reward_range`, :attr:`metadata` and :attr:`policy` attributes, without changing the underlying SCM's attributes.
    Moreover, the behavior of the :meth:`see`, :meth:`do`, :meth:`action`, and :meth:`observation' methods can be changed by these wrappers.

    Some attributes (:attr:`spec`, :attr:`render_mode`, :attr:`np_random`) will point back to the wrapper's environment
    (i.e. to the corresponding attributes of :attr:`env`).

    Note:
        If you inherit from :class:`PCHWrapper`, don't forget to call ``super().__init__(env)``
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

    @property
    def unwrapped(self,) -> Union[Env[ObsType, ActType], Generic[PolicyType, ObsType, ActType]]:
        # This will return the underlying environment without wrappers
        return self.env.unwrapped


class ObservationPCHWrapper(
    PCHWrapper[PolicyType, WrapperObsType, ActType, PolicyType, ObsType, ActType], 
):
    """Modify observations from :meth:`Env.see` and :meth:`Env.do` using :meth:`wrap_observation` function.

    If you would like to apply a function to only the observation before
    passing it to the learning code, you can simply inherit from :class:`ObservationPCHWrapper` and overwrite the method
    :meth:`wrap_observation` to implement that transformation. The transformation defined in that method must be
    reflected by the :attr:`env` observation space. Otherwise, you need to specify the new observation space of the
    wrapper by setting :attr:`self.observation_space` in the :meth:`__init__` method of your wrapper.
    """

    def __init__(self, env: SCM[PolicyType, ObsType, ActType]):
        """Constructor for the observation wrapper."""
        PCHWrapper.__init__(self, env)

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

class RewardPCHWrapper(
    PCHWrapper[PolicyType, ObsType, ActType, PolicyType, ObsType, ActType], 
):
    """Superclass of wrappers that can modify the returning reward from one stage of interaction.

    If you would like to apply a function to the reward that is returned by the base environment before
    passing it to learning code, you can simply inherit from :class:`RewardPCHWrapper` and overwrite the method
    :meth:`wrap_reward` to implement that transformation.
    This transformation might change the :attr:`reward_range`; to specify the :attr:`reward_range` of your wrapper,
    you can simply define :attr:`self.reward_range` in :meth:`__init__`.
    """
        
    def __init__(self, env: SCM[PolicyType, ObsType, ActType]):
        """Constructor for the Reward wrapper."""
        PCHWrapper.__init__(self, env)

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

class ActionPCHWrapper(
    PCHWrapper[PolicyType, ObsType, WrapperActType, PolicyType, ObsType, ActType], 
):
    """Superclass of wrappers that can modify the action before :meth:`env.do` and returned from :meth:`env.see`.

    If you would like to apply a function to the action before passing it to the base environment,
    you can simply inherit from :class:`ActionPCHWrapper` and overwrite the method  :meth:`wrap_action` and :meth:`unwrap_action` to implement
    that transformation. The transformation defined in that method must take values in the base environment’s
    action space. However, its domain might differ from the original action space.
    In that case, you need to specify the new action space of the wrapper by setting :attr:`self.action_space` in
    the :meth:`__init__` method of your wrapper.
    """

    def __init__(self, env: SCM[PolicyType, ObsType, ActType]):
        """Constructor for the action wrapper."""
        PCHWrapper.__init__(self, env)

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
    
class PolicyPCHWrapper(
    PCHWrapper[WrapperPolicyType, ObsType, ActType, PolicyType, ObsType, ActType], 
):
    """Superclass of wrappers that can modify the policy deployed in the environment.

    If you would like to deploy a policy to the base environment,
    you can simply inherit from :class:`PolicyPCHWrapper` and overwrite the method  :meth:`action` and :meth:`see` to implement
    that transformation. The policy defined in that method must take values in the base environment’s
    action space. However, its domain might differ from the original action space.
    In that case, you need to specify the new action space of the wrapper by setting :attr:`self.action_space` in
    the :meth:`__init__` method of your wrapper.
    """

    def __init__(self, env: SCM[PolicyType, ObsType, ActType]):
        """Constructor for the action wrapper."""
        PCHWrapper.__init__(self, env)

    def action(self) -> ActType:
        """Modifies the :attr:`env` :meth:`action` using the new policy"""
        raise NotImplementedError
    
    def see(self) -> tuple[WrapperActType, ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Runs the :attr:`env` :meth:`env.do` using the modified ``action`` from :meth:`self.action`."""
        raise NotImplementedError