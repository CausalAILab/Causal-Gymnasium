from __future__ import annotations
from enum import IntEnum, StrEnum
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Union
from collections import namedtuple

from .policy_scope import PolicyScope


class LearningRegime(StrEnum):
    see = "see"
    do = "do"
    ctf_do = "ctf_do"
    cool = "cool"


class Assumptions(StrEnum):
    dag = "dag"
    markov = "markov"
    nuc = "nuc"


class RewardFunc(StrEnum):
    discount = "discount"
    avg = "average"
    sum = "sum"


# Define the task as a named tuple
class Task(namedtuple('Task', ['learning_regime', 'assumptions', 'policy_space', 'reward_func'])):
    __slots__ = ()
    def __new__(
        cls,
        learning_regime: LearningRegime = LearningRegime.do,
        assumptions: Assumptions = Assumptions.dag,
        policy_space: PolicyScope = None,
        reward_func: RewardFunc = RewardFunc.discount
    ):
        assert learning_regime in LearningRegime.__members__.values() or learning_regime in LearningRegime, \
            f"{learning_regime} is not a valid LearningRegime"
        assert assumptions in Assumptions.__members__.values() or assumptions in Assumptions, \
            f"{assumptions} is not a valid Assumptions"
        assert reward_func in RewardFunc.__members__.values() or reward_func in RewardFunc, \
            f"{reward_func} is not a valid RewardFunc"
        # Leave policy scope check blank for now
        return super(Task, cls).__new__(cls, learning_regime, assumptions, policy_space, reward_func)
    

