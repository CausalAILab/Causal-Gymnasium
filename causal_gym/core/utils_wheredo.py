from typing import Iterable, Generator, Tuple, TypeVar, AbstractSet, List, Set
from itertools import combinations as itercomb
# import causal_gym as cgym
from .causal_graph import CausalGraph

T = TypeVar('T')

def combinations(xs: Iterable[T]) -> Generator[Tuple[T, ...], None, None]:
    """ all combinations of given in the order of increasing its size """
    xs = list(xs)
    for i in range(len(xs) + 1):
        for comb in itercomb(xs, i):
            yield comb

def only(W: List[T], Z: AbstractSet[T]) -> List[T]:
    if not Z:
        return []
    return [w for w in W if w in Z]


def pop(xs: Set):
    x = next(iter(xs))
    xs.remove(x)
    return x

def CC(G: CausalGraph, X: str):
    """ an X containing c-component of G  """
    return G.c_component(X)