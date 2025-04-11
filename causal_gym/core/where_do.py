from typing import Iterable, TypeVar, Generator, Tuple, Set, List, FrozenSet, AbstractSet
from itertools import combinations as itercomb
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


def MISs(G: CausalGraph, Y: str) -> FrozenSet[FrozenSet[str]]:
    """ All minimal intervention sets """
    II = G.V - {Y}
    assert II <= G.V
    assert Y not in II

    G = G[G.An(Y)]
    Ws = G.causal_order(backward=True)
    Ws = only(Ws, II)
    return subMISs(G, Y, frozenset(), Ws)


def subMISs(G: CausalGraph, Y: str, Xs: FrozenSet[str], Ws: List[str]) -> FrozenSet[FrozenSet[str]]:
    """ subroutine for MISs -- this creates a recursive call tree with n, n-1, n-2, ... widths """
    out = frozenset({Xs})
    for i, W_i in enumerate(Ws):
        H = G.do({W_i})
        H = H[H.An(Y)]
        out |= subMISs(H, Y, Xs | {W_i}, only(Ws[i + 1:], H.V))
    return out


def bruteforce_POMISs(G: CausalGraph, Y: str) -> FrozenSet[FrozenSet[str]]:
    """ This computes a complete set of POMISs in a brute-force way """
    return frozenset({frozenset(IB(G.do(Ws), Y))
                      for Ws in combinations(list(G.V - {Y}))})


def MUCT(G: CausalGraph, Y: str) -> FrozenSet[str]:
    """ Minimal Unobserved Confounder's Territory """
    
    H = G[G.An(Y)]
    # print('An',G.An(Y))
    # print('H',H.V)
    Qs = {Y}
    Ts = frozenset({Y})
    # print('pa',G._pa)
    # print('ch',G._ch)
    while Qs:
        Q1 = pop(Qs)
        Ws = CC(H, Q1)
        Ts |= Ws
        Qs = (Qs | H.de(Ws)) - Ts
        # print('Q:',Qs)
        # print('T:',Ts)

    return Ts


def IB(G: CausalGraph, Y: str) -> FrozenSet[str]:
    """ Interventional Border """
    Zs = MUCT(G, Y)
    return G.pa(Zs) - Zs


def MUCT_IB(G: CausalGraph, Y) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    Zs = MUCT(G, Y)
    return Zs, G.pa(Zs) - Zs


def POMISs(G: CausalGraph, Y: str) -> Set[FrozenSet[str]]:
    """ all POMISs for G with respect to Y """
    G = G[G.An(Y)]

    Ts, Xs = MUCT_IB(G, Y)
    H = G.do(Xs)[Ts | Xs]
    return subPOMISs(H, Y, only(H.causal_order(backward=True), Ts - {Y})) | {frozenset(Xs)}


def subPOMISs(G: CausalGraph, Y, Ws: List, obs=None) -> Set[FrozenSet[str]]:
    if obs is None:
        obs = set()

    out = []
    for i, W_i in enumerate(Ws):
        Ts, Xs = MUCT_IB(G.do({W_i}), Y)
        new_obs = obs | set(Ws[:i])
        if not (Xs & new_obs):
            out.append(Xs)
            new_Ws = only(Ws[i + 1:], Ts)
            if new_Ws:
                out.extend(subPOMISs(G.do(Xs)[Ts | Xs], Y, new_Ws, new_obs))
    return {frozenset(_) for _ in out}


def minimal_do(G: CausalGraph, Y: str, Xs: AbstractSet[str]) -> FrozenSet[str]:
    """ Non-redundant subset of Xs that entail the same E[Y|do(Xs)] """
    return frozenset(Xs & G.do(Xs).An(Y))

def IV_CD(uname='U_XY'):
    X, Y, Z = 'X', 'Y', 'Z'
    return CausalGraph({X, Y, Z}, [(Z, X), (X, Y)], [(X, Y, uname)])

def XYZWST(u_wx='U0', u_yz='U1'):
    W, X, Y, Z, S, T = 'W', 'X', 'Y', 'Z', 'S', 'T'
    return CausalGraph({'W', 'X', 'Y', 'Z', 'S', 'T'}, [(Z, X), (X, Y), (W, Y), (S, W), (T, X), (T, Y)], [(X, W, u_wx), (Z, Y, u_yz)])

def XYZW(u_wx='U0', u_yz='U1'):
    return XYZWST(u_wx, u_yz) - {'S', 'T'}
    # return XYZWST(u_wx, u_yz)['X','W','Y','Z']

def simple_markovian():
    X1, X2, Y, Z1, Z2 = 'X1', 'X2', 'Y', 'Z1', 'Z2'
    return CausalGraph({'X1', 'X2', 'Y', 'Z1', 'Z2'}, [(X1, Y), (X2, Y), (Z1, X1), (Z1, X2), (Z2, X1), (Z2, X2)])

# cdag = IV_CD()
# # cdag = cdag.do({'Z'})
# # cdag.nx_viz()
# print('POMISs', POMISs(cdag,'Y'))
# print('IB', IB(cdag, 'Y'))
# print('MUCT_IB', MUCT_IB(cdag, 'Y'))
# print('MISs:', MISs(cdag, 'Y'))
