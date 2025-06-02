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


class Assumptions(StrEnum):
    dag = "dag"
    markov = "markov"
    iid = "iid"
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
        learning_regime: LearningRegime = LearningRegime.see,
        assumptions: Assumptions = Assumptions.dag,
        policy_space: PolicyScope = None,
        reward_func: RewardFunc = RewardFunc.discount
    ):
        return super(Task, cls).__new__(cls, learning_regime, assumptions, policy_space, reward_func)
    

