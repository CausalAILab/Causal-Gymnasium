from __future__ import annotations

import numpy as np
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Union

from collections import namedtuple
from .types import *


# Define the policy scope as a named tuple
class PolicyScope(namedtuple('PolicyScope', ['act', 'obs'])):
    __slots__ = ()  # Named tuples are immutable, so we use __slots__ to prevent creating a __dict__
    def __new__(cls, act:list[ActType]=[], obs:list[ObsType]=[]):
        return super(PolicyScope, cls).__new__(cls, act, obs)
    
