import functools
from typing import (
    Tuple, FrozenSet, List, Set, AbstractSet
)
from .causal_graph import CausalGraph
from .utils_wheredo import only, combinations, pop, CC

class WhereDo:
    def __init__(self):
        self._do = functools.lru_cache(maxsize=None)(self._do_uncached)
        self._an = functools.lru_cache(maxsize=None)(self._an_uncached)
        self._order = functools.lru_cache(maxsize=None)(self._order_uncached)
        pass

    def _do_uncached(self, G, ws: Tuple[str, ...]) -> CausalGraph:
        return G.do(set(ws))

    def _an_uncached(self, G, v: str) -> FrozenSet[str]:
        return G.An(v)

    def _order_uncached(self, G, backward: bool) -> Tuple[str, ...]:
        return tuple(G.causal_order(backward=backward))

    def MISs(self, G, Y: str) -> Set[FrozenSet[str]]:
        II = frozenset(G.V - {Y})
        subG = G[self._an(Y)]
        Ws = tuple(w for w in self._order(True) if w in II)
        raw_sets = self._subMISs(Y, (), Ws, subG)
        return {frozenset(s) for s in raw_sets}

    def bruteforce_POMISs(self, G, Y: str) -> Set[FrozenSet[str]]:
        Vs = list(G.V - {Y})
        return {
            frozenset(self.IB(Y, ws))
            for ws in combinations(Vs)
        }

    def MUCT(self, G, Y: str) -> FrozenSet[str]:
        H = G[self._an(Y)]
        Ts = {Y}
        Qs = {Y}
        while Qs:
            q = pop(Qs)
            new = CC(H, q)
            Ts |= new
            Qs = (Qs | H.de(new)) - Ts
        return frozenset(Ts)

    def IB(self, G, Y: str, Ws: AbstractSet[str]=None) -> FrozenSet[str]:
        G = (self._do(tuple(Ws)) if Ws else G)[self._an(Y)]
        Zs = self.MUCT(Y) if not Ws else self.MUCT(Y)
        return frozenset(G.pa(Zs) - set(Zs))

    def MUCT_IB(self, Y: str) -> Tuple[FrozenSet[str], FrozenSet[str]]:
        Zs = self.MUCT(Y)
        Xs = frozenset(G.pa(Zs) - Zs)
        return Zs, Xs

    def POMISs(self, Y: str) -> Set[FrozenSet[str]]:

        ancG = self.G[self._an(Y)]
        Ts, X0 = self.MUCT_IB(Y)
        H = self._do(tuple(X0))[tuple(Ts | X0)]
        rest = tuple(w for w in self._order(True) if w in (Ts - {Y}))
        raw = self._subPOMISs(Y, rest, H, obs=())
        raw.add(tuple(X0))
        return {frozenset(s) for s in raw}

    def minimal_do(self, Y: str, Xs: AbstractSet[str]) -> FrozenSet[str]:
        return frozenset(Xs & set(self._an(Y)))

    # --- internal recursion helpers ---
    def _subMISs(
        self,
        Y: str,
        Xs: Tuple[str, ...],
        Ws: Tuple[str, ...],
        G: CausalGraph
    ) -> Set[Tuple[str, ...]]:
        out = {Xs}
        # prune if len(Xs) >= best_known_size: …
        for i, w in enumerate(Ws):
            newG = self._do((w,))[tuple(self._an(Y))]
            tail = tuple(w2 for w2 in Ws[i+1:] if w2 in newG.V)
            for s in self._subMISs(Y, Xs + (w,), tail, newG):
                out.add(s)
        return out

    def _subPOMISs(
        self,
        Y: str,
        Ws: Tuple[str, ...],
        G: CausalGraph,
        obs: Tuple[str, ...]
    ) -> Set[Tuple[str, ...]]:
        out = set()
        for i, w in enumerate(Ws):
            Ts, Xs = self.MUCT_IB(Y)
            if set(Xs) & set(obs):
                continue
            out.add(tuple(Xs))
            newWs = tuple(w2 for w2 in Ws[i+1:] if w2 in Ts)
            if newWs:
                out |= self._subPOMISs(Y, newWs, G, obs + (w,))
        return out