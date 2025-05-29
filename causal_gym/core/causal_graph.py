from __future__ import annotations

import networkx as nx
import numpy as np
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar, Union, Optional, Iterable, Tuple
from typing import Set, FrozenSet, Dict, Sequence, AbstractSet


import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.patches import FancyArrowPatch
import numpy as np

import itertools
from itertools import product, chain
from collections import defaultdict
import functools

from collections import namedtuple
from gymnasium import Env, Wrapper
if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec, WrapperSpec


class CausalGraph:
    '''
    Description:
    Causal DAG object invoked to create, compute ancestors, descendants, c-components, intervene .etc on causal graphs.

    To perfom an intervention: Instantiate CausalGraph without Do first, i.e only (V, DE, BDE). Then invoke meth: .do
    To slice subgraph preserving directed edges:...

    Input: (CausalGraph, Do, Ind) or (V, DE, BDE)
    V: Iterable[str] - Endogenous variable names in causal graph
    DE: list[Tuple[V_i: str, V_j str]] - Direct edges mapping from V_i to V_j
    BDE: list[Tuple[ V_i: str, V_j: str, U_ij: str]] - Bidirected edges mapping from V_i to V_j via U_ij
    Do: Set[str] - Set of intervened variables
    Ind: Set[str] - Slice/Subgraph of G with the direct edges preserved
    
    '''
    def __init__(self, 
                 V: Optional[Iterable[str]], 
                 DE: Optional[list[Tuple[str, str]]] = frozenset(), 
                 BDE: Optional[list[Tuple[str, str, str]]] = frozenset(),
                 cdag: 'CausalGraph' = None, 
                 Do: Optional[Set[str]] = None,
                 Ind: Optional[Set[str]] = None
                 ):
        with_do = cover(Do)
        with_induct = cover(Ind)
        #print(with_do, V)
        if cdag is not None:
            if with_do is not None:
                self.V = cdag.V
                self.U = cover(u for u in cdag.U if with_do.isdisjoint(cdag.confounded_dict[u]))
                #print(V)
                self.confounded_dict = {u: val for u, val in cdag.confounded_dict.items() if u in self.U}

                dopa = cdag.pa(with_do)
                doAn = cdag.An(with_do)
                doDe = cdag.De(with_do)

                self._pa = defaultdict(frozenset, {k: frozenset() if k in with_do else v for k, v in cdag._pa.items()})
                self._ch = defaultdict(frozenset, {k: v - with_do if k in dopa else v for k, v in cdag._ch.items()})
                self._an = dict_except(cdag._an, doDe)
                self._de = dict_except(cdag._de, doAn)
                #print(self._pa,' $$ ', self._ch)
            elif with_induct is not None:
                assert with_induct <= cdag.V
                removed = cdag.V - with_induct
                self.V = with_induct
                self.confounded_dict = {u: val for u, val in cdag.confounded_dict.items() if val <= self.V} #
                self.U = cover(self.confounded_dict)

                removed_ch = cdag.pa(removed) & self.V
                removed_pa = cdag.ch(removed) & self.V
                removed_an = cdag.de(removed) & self.V
                removed_de = cdag.an(removed) & self.V

                
                self._pa = defaultdict(frozenset, {x: (cdag._pa[x]-removed) if x in removed_pa else cdag._pa[x] for x in self.V})
                self._ch = defaultdict(frozenset, {x: (cdag._ch[x]-removed) if x in removed_ch else cdag._ch[x] for x in self.V})
                self._an = dict_only(cdag._an, self.V - removed_an)
                self._de = dict_only(cdag._de, self.V - removed_de)
                #print(self._pa,' $$ ', self._ch)

            else:
                self.V = cdag.V
                self.U = cdag.U
                self.confounded_dict = cdag.confounded_dict
                self._ch = cdag._ch
                self._pa = cdag._pa
                self._an = cdag._an
                self._de = cdag._de
        else:
            # Initialize causal graph elements
            self.V = frozenset(V) | fs_union(DE) | frozenset({x for y in BDE for x in y[:2]})
            self.U = frozenset({bde[2] if len(bde) < 3 else 'u_' + ''.join(sorted([bde[0], bde[1]])) for bde in BDE})
            self.confounded_dict = {e[2] if len(e) < 3 else 'u_' + ''.join(sorted([e[0], e[1]])): frozenset({e[0], e[1]}) for e in BDE}   

            bde_dict = defaultdict(set)
            for e in DE:
                bde_dict[e[0]].add(e[1])
            self._ch = defaultdict(frozenset, {key: frozenset(vals) for key, vals in bde_dict.items()})
            bde_dict = defaultdict(set)
            for e in DE:
                bde_dict[e[1]].add(e[0])
            self._pa = defaultdict(frozenset, {key: frozenset(vals) for key, vals in bde_dict.items()})
            self._an = dict()
            self._de = dict()
            assert self._ch.keys() < self.V and self._pa.keys() < self.V # |ch| < |V|, |pa| < |V|

        self.edges = tuple((x, ch) for x, chs in self._ch.items() for ch in chs)
        self.causal_order = functools.lru_cache()(self.causal_order)
        self._do_ = functools.lru_cache()(self._do_)
        self.__cc = None
        self.__cc_dict = None
        self.__h = None
        self.__characteristic = None
        self.__confoundeds = None
        self.u_pas = defaultdict(set)
        for u, xy in self.confounded_dict.items():
            for v in xy:
                self.u_pas[v].add(u)
        self.u_pas = defaultdict(set, {v: frozenset(us) for v, us in self.u_pas.items()})
        #print(self._pa)

        #print(self.confounded_dict)
        #print(self.V)
        #print(self.edges)
        #print("##"*10)
        pass

    def Us(self, V):
        return self.u_pas[V]
    
    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.V or item in self.U
        if len(item) == 2:
            if isinstance(item, AbstractSet):
                x, y = item
                return self.is_confounded(x, y)
            else:
                return tuple(item) in self.edges
        elif len(item) == 3:
            x, y, u = item
            return self.is_confounded(x, y) and u in self.confounded_dict and self.confounded_dict[u] == frozenset({x, y})
    
    def __lt__(self, other):
        if not isinstance(other, CausalGraph):
            return False
        return self <= other and self != other
    def __gt__(self, other):
        if not isinstance(other, CausalGraph):
            return False
        return self >= other and self != other
    
    def __le__(self, other):
        if not isinstance(other, CausalGraph):
            return False
        return self.V <= other.V and set(self.edges) <= set(other.edges) and set(self.confounded_dict.values()) <= set(other.confounded_dict.values())
    def __ge__(self, other):
        if not isinstance(other, CausalGraph):
            return False
        return self.V >= other.V and set(self.edges) >= set(other.edges) and set(self.confounded_dict.values()) >= set(other.confounded_dict.values())
    
    def Pa(self, v) -> FrozenSet:
        return self.pa(v) | cover(v)
    def pa(self, v) -> FrozenSet:
        if isinstance(v, str):
            return self._pa[v]
        else:
            return fs_union(self._pa[v_i] for v_i in v)
    
    def Ch(self, v) -> FrozenSet:
        return self.ch(v) | cover(v, frozenset)
    def ch(self, v) -> FrozenSet:
        if isinstance(v, str):
            return self._ch[v]
        else:
            return fs_union(self._ch[v_i] for v_i in v)
    
    def An(self, v) -> FrozenSet:
        return self.an(v) | cover(v, frozenset)
    def an(self, v) -> FrozenSet:
        if isinstance(v, str):
            return self.get_an(v)
        else:
            return fs_union(self.get_an(v_i) for v_i in v)
    
    def De(self, v) -> FrozenSet:
        return self.de(v) | cover(v, frozenset)
    def de(self, v) -> FrozenSet:
        if isinstance(v, str):
            return self.get_de(v)
        else:
            return fs_union(self.get_de(v_i) for v_i in v)
        
    def get_an(self, v) -> FrozenSet:
        if v in self._an:
            return self._an[v]
        # self._an[v] = fs_union(self._an(pa) for pa in self._pa[v]) | self._pa[v]
        self._an[v] = fs_union(self.get_an(parent) for parent in self._pa[v]) | self._pa[v]
        # self._an[v] = frozenset()
        # stack = [v]
        # while stack:
        #     pa = stack.pop()
        #     self._an[v] |= self._pa[pa]
        #     stack.extend(self._pa[pa] - self._an[v])
        return self._an[v]
    def get_de(self, v) -> FrozenSet:
        if v in self._de:
            return self._de[v]
        # self._de[v] = fs_union(self._de(ch) for ch in self._ch[v]) | self._ch[v]
        self._de[v] = fs_union(self.get_de(child) for child in self._ch[v]) | self._ch[v]
        # self._de[v] = frozenset()
        # stack = [v]
        # while stack:
        #     ch = stack.pop()
        #     self._de[v] |= self._ch[ch]
        #     stack.extend(self._ch[ch] - self._de[v])
        return self._de[v]
    
    def do(self, v) -> CausalGraph:
        return self._do_(cover(v))
    def _do_(self, v) -> CausalGraph:
        return CausalGraph(None, None, None, self, cover(v))

    def has_edge(self, x, y) -> bool:
        return y in self._ch[x]
    def is_confounded(self, x, y) -> bool:
        return {x, y} in self.confounded_dict.values()
    
    def u_of(self, x, y):
        key = {x, y}
        for u, ab in self.confounded_dict.items():
            if ab == key:
                return u
        return None
    
    def confounded_with(self, u):
        return self.confounded_dict[u]
    def confounded_withs(self, v):
        return {next(iter(xy - {v})) for xy in self.confounded_dict.values() if v in xy}

    def __getitem__(self, item) -> CausalGraph:
        return self.induced(item)

    def induced(self, v_or_vs) -> CausalGraph:
        if set(v_or_vs) == self.V:
            return self
        return CausalGraph(None, None, None, cdag=self, Ind=v_or_vs)
    

    @property
    def characteristic(self):
        if self.__characteristic is None:
            self.__characteristic = (len(self.V),
                                     len(self.edges),
                                     len(self.confounded_dict),
                                     sortup([(len(self.ch(v)), len(self.pa(v)), len(self.confounded_withs(v))) for v in self.V]))
        return self.__characteristic

    def edges_removed(self, edges_to_remove: Iterable[Sequence[str]]) -> CausalGraph:
        # edges_to_remove = [tuple(edge) for edge in edges_to_remove]
        dir_edges, bidir_edges = set(), set()
        for edge in edges_to_remove:
            if len(edge) == 2:
                dir_edges.add(tuple(edge))
            elif len(edge) == 3:
                bidir_edges.add((*sorted(edge[:2]), edge[2]))
        return CausalGraph(self.V, set(self.edges) - dir_edges, self.confounded_to_3tuples() - frozenset(bidir_edges))
        # dir_edges = {edge for edge in edges_to_remove if len(edge) == 2}
        # bidir_edges = {edge for edge in edges_to_remove if len(edge) == 3}
        # bidir_edges = frozenset((*sorted([x, y]), u) for x, y, u in bidir_edges)
        # return CausalGraph(self.V, set(self.edges) - dir_edges, self.confounded_to_3tuples() - bidir_edges)

    def __sub__(self, v_or_vs_or_edges) -> CausalGraph:     
        if not v_or_vs_or_edges:
            return self
        if isinstance(v_or_vs_or_edges, str):
            return self[self.V - cover(v_or_vs_or_edges)]
        v_or_vs_or_edges = list(v_or_vs_or_edges)
        if isinstance(v_or_vs_or_edges[0], str):
            return self[self.V - cover(v_or_vs_or_edges)]
        return self.edges_removed(v_or_vs_or_edges)

    def causal_order(self, backward=False) -> Tuple:    
        gg = nx.DiGraph(self.edges)
        gg.add_nodes_from(self.V)
        top_to_bottom = list(nx.topological_sort(gg))
        if backward:
            return tuple(reversed(top_to_bottom))
        else:
            return tuple(top_to_bottom)

    def __add__(self, edges):
        if isinstance(edges, CausalGraph):
            return merge_two_cds(self, edges)

        directed_edges = {edge for edge in edges if len(edge) == 2}
        bidirected_edges = {edge for edge in edges if len(edge) == 3}
        return CausalGraph(self.V, set(self.edges) | directed_edges, self.confounded_to_3tuples() | bidirected_edges)

    def __ensure_confoundeds_cached(self):
        if self.__confoundeds is None:
            self.__confoundeds = dict()
            for u, (x, y) in self.confounded_dict.items():
                if x not in self.__confoundeds:
                    self.__confoundeds[x] = set()
                if y not in self.__confoundeds:
                    self.__confoundeds[y] = set()
                self.__confoundeds[x].add(y)
                self.__confoundeds[y].add(x)
            self.__confoundeds = {x: frozenset(ys) for x, ys in self.__confoundeds.items()}
            for v in self.V:
                if v not in self.__confoundeds:
                    self.__confoundeds[v] = frozenset()

    def __ensure_cc_cached(self):
        if self.__cc is None:
            self.__ensure_confoundeds_cached()
            ccs = []
            remain = set(self.V)
            found = set()
            while remain:
                v = next(iter(remain))
                a_cc = set()
                to_expand = [v]
                while to_expand:
                    v = to_expand.pop()
                    a_cc.add(v)
                    # to_expand += list(self.__confoundeds[v] - a_cc)
                    to_expand.extend(self.__confoundeds[v] - a_cc)
                ccs.append(a_cc)
                found |= a_cc
                remain -= found
            self.__cc2 = frozenset(frozenset(a_cc) for a_cc in ccs)
            self.__cc_dict2 = {v: a_cc for a_cc in self.__cc2 for v in a_cc}

            self.__cc = self.__cc2
            self.__cc_dict = self.__cc_dict2

    @property
    def c_components(self) -> FrozenSet:
        self.__ensure_cc_cached()
        return self.__cc

    def c_component(self, v_or_vs) -> FrozenSet:
        assert isinstance(v_or_vs, str)
        self.__ensure_cc_cached()
        return fs_union(self.__cc_dict[v] for v in cover(v_or_vs))

    def confounded_to_3tuples(self) -> FrozenSet[Tuple[str, str, str]]:
        return frozenset((*sorted([x, y]), u) for u, (x, y) in self.confounded_dict.items())

    def __eq__(self, other):
        if not isinstance(other, CausalGraph):
            return False
        if self.V != other.V:
            return False
        if set(self.edges) != set(other.edges):
            return False
        if set(self.confounded_dict.values()) != set(other.confounded_dict.values()):  # does not care about U's name
            return False
        return True

    def __hash__(self):
        if self.__h is None:
            self.__h = hash(sortup(self.V)) ^ hash(sortup(self.edges)) ^ hash(sortup2(self.confounded_dict.values()))
        return self.__h
    
    def nx_viz(
        self,
        path: str = "./cdag.png",
        node_color_map: Optional[Dict[str, Set[str]]] = None,
        pos: Optional[Dict[str, Tuple[float, float]]] = None,
        labels: Optional[Dict[str, str]] = None,
        *,
        # --- New parameters for MUCT ring ------------------------------------------------
        muct_nodes: Optional[Set[str]] = None,
        ring_color: str = "lightsteelblue",  # light blue ring colour
        ring_scale: float = 1.35,             # how much larger than the base node the ring is
        ring_width: float = 6.0,              # thickness of the ring in points
        node_size: int = 300  # Default node size
    ):
        """
        Creates a better-arranged NX image with optional colored node groups.
        MUCT ring parameters are experimental and may change.

        Parameters
        ----------
        path : str
            Output path for the saved figure.
        node_color_map : Dict[str, Set[str]], optional
            Mapping from *fill colour* → nodes to be filled with that colour.
        pos : Dict[str, Tuple[float, float]], optional
            Fixed positions for the nodes (passed directly to NetworkX).
        labels : Dict[str, str], optional
            Custom labels for nodes. Falls back to the node names.
        muct_nodes : Set[str], optional (new)
            Nodes to highlight with a light‑blue outer ring.  If ``None`` the function
            attempts a best‑effort guess using ``node_color_map`` (looks for any colour
            containing the substring *"lightblue"*).
        ring_color / ring_scale / ring_width
            Appearance controls for the ring.
        node_size : int, optional
            Default node size for drawing.
        """
        # ------------------------------------------------------------------
        # Build NetworkX graph ---------------------------------------------------------
        self.__ensure_confoundeds_cached()
        G = nx.DiGraph()
        G.add_nodes_from(self.V)
        G.add_edges_from(self.edges)

        # Layout ----------------------------------------------------------------------
        if pos is None:
            try:
                pos = graphviz_layout(G, prog="dot")
            except Exception:  # fall back to spring layout if GraphViz not available
                pos = nx.spring_layout(G, seed=42)

        # ------------------------------------------------------------------
        # Colour map handling ----------------------------------------------------------
        default_color = "white"
        node_colors = {node: default_color for node in self.V}
        if node_color_map:
            for colour, nodes in node_color_map.items():
                for node in nodes:
                    if node in node_colors:
                        node_colors[node] = colour
        node_color_list = [node_colors[node] for node in G.nodes]

        # ------------------------------------------------------------------
        # Attempt to infer MUCT nodes automatically if not provided --------------------
        if muct_nodes is None and node_color_map:
            for colour_key, nodes in node_color_map.items():
                if "lightblue" in colour_key.lower():
                    muct_nodes = set(nodes)
                    break
        if muct_nodes is None:
            muct_nodes = set()

        # ------------------------------------------------------------------
        # Plot -------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        # Draw nodes (base)
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_color_list,
            node_size=node_size,
            edgecolors="black",
            linewidths=1.5,
            ax=ax,
        )

        # Draw labels
        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            font_size=12,
            font_weight="bold",
            ax=ax,
        )

        # Draw directed edges
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=self.edges,
            arrowstyle="-|>",
            arrowsize=20,
            edge_color="black",
            node_size=node_size,
            ax=ax,
        )

        # Draw bidirected (confounding) edges -----------------------------------------
        for u, (x, y) in self.confounded_dict.items():
            if x in pos and y in pos:
                dx, dy = pos[y][0] - pos[x][0], pos[y][1] - pos[x][1]
                dist = np.hypot(dx, dy)
                if dist == 0:
                    continue
                start = (pos[x][0] + 0.0 * dx, pos[x][1] + 0.0 * dy)
                end = (pos[y][0] - 0.0 * dx, pos[y][1] - 0.0 * dy)
                arrow = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-",
                    linestyle=(0, (5, 10)),
                    color="gray",
                    connectionstyle="arc3,rad=0.2",
                    linewidth=1.5,
                )
                ax.add_patch(arrow)

        # ------------------------------------------------------------------
        # *NEW*   Draw light‑blue ring around MUCT nodes -------------------------------
        if muct_nodes:
            overlay_node_size = node_size * ring_scale
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=list(muct_nodes),
                node_color="none",           # transparent fill
                node_size=overlay_node_size, # larger size → outer ring
                edgecolors=ring_color,
                linewidths=ring_width,
                ax=ax,
            )

        # Final touches ----------------------------------------------------------------
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return (fig, G)


        
T = TypeVar('T')
# def fs_union(obj) -> FrozenSet:
#     return frozenset() if not obj else frozenset(set.union(*obj))
def fs_union(sets) -> FrozenSet:
    return frozenset(chain(*sets))
def cover(obj, _with=frozenset):
    return None if obj is None else (_with({obj}) if isinstance(obj, str) else _with(obj))
def sortup(xs: Iterable[T]) -> Tuple[T, ...]:
    return tuple(sorted(xs))
def sortup2(xxs):
    return sortup([sortup(xs) for xs in xxs])
def disjoint(set1: Set, set2: Set) -> bool:
    if len(set2) < len(set1):
        set1, set2 = set2, set1
    return not any(x in set2 for x in set1)
def dict_only(a_dict: dict, keys: AbstractSet) -> Dict:
    return {k: a_dict[k] for k in keys if k in a_dict}
def dict_except(a_dict: dict, keys: AbstractSet) -> Dict:
    return {k: v for k, v in a_dict.items() if k not in keys}
def merge_two_cds(g1: CausalGraph, g2: CausalGraph) -> CausalGraph:
    VV = g1.V | g2.V
    EE = set(g1.edges) | set(g2.edges)
    VWU = set(g1.confounded_to_3tuples()) | set(g2.confounded_to_3tuples())
    return CausalGraph(VV, EE, VWU)
def with_default(x, dflt=None):
    return x if x is not None else dflt
def default_P_U(mu: Dict):
    def P_U(d):
        p_val = 1.0
        # print(mu)
        for k in mu.keys():
            p_val *= (1 - mu[k]) if d[k] == 0 else mu[k]
        return p_val

    return P_U
def flatten(input_data):
    result = set()
    
    def recurse(item):
        if isinstance(item, (set, frozenset)):
            for elem in item:
                recurse(elem)
        elif isinstance(item, (list, tuple)):
            for elem in item:
                recurse(elem)
        elif isinstance(item, str):
            result.add(item)
    
    recurse(input_data)
    return result


# Test CDAG
# def IV_CD(uname='U_XY'):
#     X, Y, Z = 'X', 'Y', 'Z'
#     return CausalGraph({X, Y, Z}, [(Z, X), (X, Y)], [(X, Y, uname)])

# def XYZWST(u_wx='U0', u_yz='U1'):
#     W, X, Y, Z, S, T = 'W', 'X', 'Y', 'Z', 'S', 'T'
#     return CausalGraph({'W', 'X', 'Y', 'Z', 'S', 'T'}, [(Z, X), (X, Y), (W, Y), (S, W), (T, X), (T, Y)], [(X, W, u_wx), (Z, Y, u_yz)])

# def XYZW(u_wx='U0', u_yz='U1'):
#     return XYZWST(u_wx, u_yz) - {'S', 'T'}
#     # return XYZWST(u_wx, u_yz)['X','W','Y','Z']

# def simple_markovian():
#     X1, X2, Y, Z1, Z2 = 'X1', 'X2', 'Y', 'Z1', 'Z2'
#     return CausalGraph({'X1', 'X2', 'Y', 'Z1', 'Z2'}, [(X1, Y), (X2, Y), (Z1, X1), (Z1, X2), (Z2, X1), (Z2, X2)])

# cdag = simple_markovian()
# # cdag = cdag.do({'Z'})
# cdag.nx_viz()


class StructuralCausalModel:
    def __init__(self, G: CausalGraph, F=None, P_U=None, D=None, more_U=None):
        self.G = G
        self.F = F
        self.P_U = P_U
        self.D = with_default(D, defaultdict(lambda: (0, 1)))
        self.more_U = set() if more_U is None else set(more_U)

        self.query00 = functools.lru_cache(1024)(self.query00)

    def query(self, outcome: Tuple, condition: dict = None, intervention: dict = None, verbose=False) -> defaultdict:
        if condition is None:
            condition = dict()
        if intervention is None:
            intervention = dict()
        new_condition = tuple(sorted([(x, y) for x, y in condition.items()]))
        new_intervention = tuple(sorted([(x, y) for x, y in intervention.items()]))
        return self.query00(outcome, new_condition, new_intervention, verbose)

    def query00(self, outcome: Tuple, condition: Tuple, intervention: Tuple, verbose=False) -> defaultdict:
        condition = dict(condition)
        intervention = dict(intervention)

        prob_outcome = defaultdict(lambda: 0)

        U = list(sorted(self.G.U | self.more_U))
        D = self.D
        P_U = self.P_U
        V_ordered = self.G.causal_order()
        if verbose:
            print(f"ORDER: {V_ordered}")
        normalizer = 0

        for u in product(*[D[U_i] for U_i in U]):  # d^|U|
            assigned = dict(zip(U, u))
            # print('ass:',assigned)
            p_u = P_U(assigned)
            if p_u == 0:
                continue
            # evaluate values
            for V_i in V_ordered:
                if V_i in intervention:
                    assigned[V_i] = intervention[V_i]
                else:
                    assigned[V_i] = self.F[V_i](assigned)  # pa_i including unobserved

            if not all(assigned[V_i] == condition[V_i] for V_i in condition):
                continue
            normalizer += p_u
            prob_outcome[tuple(assigned[V_i] for V_i in outcome)] += p_u

        if prob_outcome:
            # normalize by prob condition
            return defaultdict(lambda: 0, {k: v / normalizer for k, v in prob_outcome.items()})
        else:
            return defaultdict(lambda: np.nan)