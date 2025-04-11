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

from .obj_types import *

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
    def get_graph(self,) :#-> tuple[dict[int, str], list[list[int]], list[list[int]]]:
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