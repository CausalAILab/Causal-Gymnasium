from __future__ import annotations

import numpy as np
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Union

from collections import namedtuple

from .policy_scope import PolicyScope

# Define the task as a named tuple
class Task(namedtuple('Task', ['learning_regime', 'str_assump', 'policy_space', 'reward_func'])):
    __slots__ = ()  # Named tuples are immutable, so we use __slots__ to prevent creating a __dict__
    def __new__(cls, learning_regime:str='see', str_assump:str='dag', policy_space:PolicyScope=None, reward_func:str='discount'):
        return super(Task, cls).__new__(cls, learning_regime, str_assump, policy_space, reward_func)
    

