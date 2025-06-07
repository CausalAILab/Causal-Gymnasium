from enum import Enum
from toposort import toposort_flatten

from .graph_defs import basicNodeType, latentNodeType, directedEdgeType, bidirectedEdgeType, undirectedEdgeType

from .set_utils import SetUtils as su
from .object_utils import ObjectUtils as ou

distribution_ids = 'distribution_ids'
suffix = '\''


class Direction(Enum):
    both = 0
    forward = 1
    backward = 2


def sortByName(node):
    return node['name']


def compareNames(n1, n2):
    return n1['name'] == n2['name']


class GraphUtils():

    # @staticmethod
    # def create():
    #     return Graph(name = 'unnamed')

    # Graph, Graph
    # boolean
    @staticmethod
    def equals(G1, G2):
        # fix edge comparison
        if su.equals(G1.nodes, G2.nodes, 'name') and len(G1.edges) == len(G2.edges):
            return True

        if len(G1.nodes) != len(G2.nodes) or len(G1.edges) != len(G2.edges):
            return False

        compNodes = dict()

        for node in G1.nodes:
            compNodes[node['name']] = True

        for node in G2.nodes:
            if node['name'] not in compNodes:
                return False

        compEdges = dict()

        for edge in G1.edges:
            compEdges[edge['from_'] + edge['type_'] + edge['to_']] = True

            # bidirected edge (X <-> Y and Y <-> X should be treated as equal)
            if edge['type_'] == bidirectedEdgeType.id_:
                compEdges[edge['to_'] + edge['type_'] + edge['from_']] = True

        for edge in G2.edges:
            if edge['from_'] + edge['type_'] + edge['to_'] not in compEdges:
                return False

        return True

    # Graph

    @staticmethod
    def clone(G, simplified=False):
        return G.copy()

#     static clone(G: Graph, simplified: boolean = false): Graph {
#         let graph: Graph = this.create();

#         for (let node of G.nodes) {
#             let newNode: Node = this.duplicateNode(node, simplified);

#             graph.nodes.push(newNode);
#         }

#         for (let edge of G.edges) {
#             let from: Node = this.getNodeByName(edge.from, graph);
#             let to: Node = this.getNodeByName(edge.to, graph);

#             if (!from || !to) continue;

#             // if non-simplified, check ids
#             if (!simplified && (!from.id || !to.id)) continue;

#             let newEdge: Edge = this.duplicateEdge(edge, from.id, to.id, simplified);

#             graph.edges.push(newEdge);
#         }

#         if (G.name)
#             graph.name = G.name;

#         if (G.task)
#             graph.task = this.duplicateTask(G.task);

#         if (G.metadata) {
#             graph.metadata = ObjectUtils.clone(G.metadata);

#             if (G.metadata.annotations)
#                 this.copyAnnotations(graph, G);
#         }

#         if (G[distribution_ids])
#             graph[distribution_ids] = ObjectUtils.clone(G[distribution_ids]);

#         return graph;
#     }

    @staticmethod
    def clear(G):
        G.nodes = []
        G.edges = []
        G.metadata = dict()

#     static clear(G: Graph) {
#         G.nodes = [];
#         G.edges = [];
#         G.metadata = Object.create(null);

#         if (G.task) {
#             for (let setType in G.task.sets) {
#                 G.task.sets[setType] = [];
#             }
#         }
#     }

#     static isSubgraph(S: Graph, G: Graph): boolean {
#         return su.isSubset(S.nodes, G.nodes) && su.isSubset(S.edges, G.edges);
#     }

#     static isMarkovian(G: Graph): boolean {
#         return G && G.edges && G.edges.filter(e => e.type == directedEdgeType.id).length == G.edges.length;
#     }

    # Node[]

    @staticmethod
    def parents(w, G):
        return G.parents(w)

    # Node[]

    @staticmethod
    def parentsPlus(w, G):
        w = ou.makeArray(w)
        return su.union(w, GraphUtils.parents(w, G))

    # Node[]

    @staticmethod
    def children(w, G):
        return G.children(w)

    # Node[]

    @staticmethod
    def neighbors(w, G, edgeType=None):
        return G.neighbors(w, edgeType)

    # Node[]

    @staticmethod
    def ancestors(w, G):
        # return G.ancestors(w)
        return GraphUtils.reach(w, G, directedEdgeType, Direction.backward)
    
    @staticmethod
    def ancestorsPlus(w, G):
        w = ou.makeArray(w)
        return su.union(w, GraphUtils.ancestors(w, G))

    # Node | Node[], Graph
    # Node[]

    @staticmethod
    def descendants(w, G):
        # return G.descendants(w)
        return GraphUtils.reach(w, G, directedEdgeType, Direction.forward)

    @staticmethod
    def descendantsPlus(w, G):
        w = ou.makeArray(w)
        return su.union(w, GraphUtils.descendants(w, G))

    # Node[]

    @staticmethod
    def nonDescendants(w, G):
        return su.difference(G.nodes, GraphUtils.descendants(w, G), 'name')

    @staticmethod
    def spouses(w, G):
        w = ou.makeArray(w)
        Sp = []

        for node in w:
            Spnode = GraphUtils.neighbors(node, G, bidirectedEdgeType)
            Sp = su.union(Sp, Spnode, 'name')
        
        return Sp

    # Graph, Node[], Node[]
    # Node[]

    @staticmethod
    def pcp(G, X, Y):
        GXbar = GraphUtils.transform(G, X, None)
        De = GraphUtils.descendants(X, GXbar)
        DeXGxbmX = su.difference(De, X, 'name')
        AnYGxb = GraphUtils.ancestors(Y, GraphUtils.transform(G, None, X))

        return su.intersection(DeXGxbmX, AnYGxb, 'name')

    # Graph, Node[], Node[]
    # Node[]

    @staticmethod
    def Dpcp(G, X, Y):
        pcp = GraphUtils.pcp(G, X, Y)
        dpcp = GraphUtils.descendants(pcp, G)

        return dpcp

    # Node | Node[], Graph, EdgeType, Direction
    # Node[]

    @staticmethod
    def reach(w, G, edgeType=None, direction=Direction.both):
        w = ou.makeArray(w)

        edges = []

        if not edgeType:
            edges = G.edges
        else:
            edges = list(filter(lambda e: e['type_'] == edgeType.id_, G.edges))

        fringe = []
        visited = dict()

        for node in w:
            fringe.append(node['name'])

        while len(fringe) > 0:
            nodeName = fringe.pop()

            # avoid cycles
            if nodeName in visited:
                continue

            visited[nodeName] = True

            for edge in edges:
                if (
                    (direction == Direction.both or direction == Direction.backward)
                    and edge['to_'] == nodeName
                    and edge['from_'] not in visited
                ):
                    fringe.append(edge['from_'])

                if (
                    (direction == Direction.both or direction == Direction.forward)
                    and edge['from_'] == nodeName
                    and edge['to_'] not in visited
                ):
                    fringe.append(edge['to_'])

        reachableNodes = list(filter(lambda n: n['name'] in visited, G.nodes))

        return reachableNodes

    # Graph, str[]
    # Node[]

    @staticmethod
    def findSimplicialVertices(G, C):
        return []

#     static findSimplicialVertices(G: Graph, C: string[]): Node[] {
#         let SV: Node[] = [];

#         for (let name of C) {
#             let node: Node = GraphUtils.getNodeByName(name, G);

#             if (!node) continue;

#             let neighbors: Node[] = GraphUtils.neighbors(node, G);
#             let pairs: Node[][] = ObjectUtils.pairs(neighbors);

#             for (let pair of pairs) {
#                 let a: Node = pair[0];
#                 let b: Node = pair[1];
#                 let aNeighbors: Node[] = GraphUtils.neighbors(a, G);
#                 if (aNeighbors.includes(b))
#                     SV.push(GraphUtils.getNodeByName(name, G));
#             }
#         }

#         return SV;
#     }

    # Node[]

    @staticmethod
    def filterBasicNodes(nodes, excludeLatentNodes=True):
        if not nodes:
            return []

        # add clusterNodeType
        if excludeLatentNodes:
            return list(filter(lambda n: n['type_'] == basicNodeType.id_, nodes))
        else:
            return list(filter(lambda n: n['type_'] == basicNodeType.id_ or n['type_'] == latentNodeType.id_, nodes))

    # Edge[]

    @staticmethod
    def getIncoming(w, G):
        w = ou.makeArray(w)
        wMap = GraphUtils.mapNodeNames(w)
        incoming = list(filter(lambda e: (
            e['type_'] == bidirectedEdgeType.id_ and e['from_'] in wMap) or e['to_'] in wMap, G.edges))

        return incoming

    # Edge[]

    @staticmethod
    def getOutgoing(w, G):
        w = ou.makeArray(w)
        wMap = GraphUtils.mapNodeNames(w)
        outgoing = list(filter(
            lambda e: e['type_'] != bidirectedEdgeType.id_ and e['from_'] in wMap, G.edges))

        return outgoing

    # Graph

    @staticmethod
    def subgraph(G, subset, activeEdges=[]):
        if G is None:
            return None

        if subset is None:
            return G

        graph = G.copy()
        graph.deleteNodes(graph.nodes)
        graph.addNodes(su.intersection(G.nodes, subset, 'name'))

        sub = dict()

        for node in subset:
            sub[node['name']] = True

        edges = list(filter(lambda e: e['from_']
                     in sub and e['to_'] in sub, G.edges))

        # if activeEdges:
        #     edges = list(filter(lambda e: e['from_'] == ))

        graph.deleteEdges(graph.edges)
        graph.addEdges(edges)

        return graph

    # Graph, Node[] | str[], Node[] | str[]

    @staticmethod
    def transform(G, overline=[], underline=[], overlineExcept=[], underlineExcept=[]):
        # if overline and len(overline) > 0 and isinstance(overline[0], str):
        #     overline = GraphUtils.getNodesByName(overline, G)

        # if underline and len(underline) > 0 and isinstance(underline[0], str):
        #     underline = GraphUtils.getNodesByName(underline, G)

        if not overline:
            overline = []
        if not underline:
            underline = []

        over = dict()
        under = dict()
        overExcept = dict()
        underExcept = dict()

        for node in overline:
            over[node['name']] = True
        for node in underline:
            under[node['name']] = True
        for node in overlineExcept:
            overExcept[node['name']] = True
        for node in underlineExcept:
            underExcept[node['name']] = True

        graph = G.copy()

        edgesToRemove = []

        for edge in graph.edges:
            if edge['to_'] in over and edge['from_'] not in overExcept:
                edgesToRemove.append(edge)
                continue

            if (
                (edge['type_'] == directedEdgeType.id_ and edge['from_']
                 in under and edge['to_'] not in underExcept)
                or (edge['type_'] == bidirectedEdgeType.id_ and edge['from_'] in over)
            ):
                edgesToRemove.append(edge)
                continue

        graph.deleteEdges(edgesToRemove)

        return graph

    # Graph, Node[], Node[]
    # Graph

    @staticmethod
    def gpbd(G, X, Y):
        X = ou.makeArray(X)
        Y = ou.makeArray(Y)
        Dpcp = GraphUtils.Dpcp(G, X, Y)

        xNames = GraphUtils.nodeToNameMap(X)
        yNames = GraphUtils.nodeToNameMap(Y)
        DpcpNames = GraphUtils.nodeToNameMap(Dpcp)

        graph = G.copy()

        edgesToRemove = []

        for edge in graph.edges:
            if edge['type_'] == directedEdgeType.id_:
                if edge['from_'] in xNames:
                    if edge['to_'] in DpcpNames or edge['to_'] in yNames:
                        edgesToRemove.append(edge)

        graph.deleteEdges(edgesToRemove)

        return graph

    # Graph
    # Graph

    @staticmethod
    def moralize(G):
        graph = G.copy()

        # convert all edges to undirected
        graph.toUndirected()

        for edge in graph.edges:
            edge['type_'] = undirectedEdgeType.id_

        edgesToAdd = []

        for node in G.nodes:
            Pa = GraphUtils.parents(node, G)

            # generate pairs
            names = list(map(lambda n: n['name'], Pa))
            pairs = [(a, b) for a in names for b in names if b > a]

            for (a, b) in pairs:
                if GraphUtils.hasEdge(a, b, graph) or GraphUtils.hasEdge(b, a, graph):
                    continue

                edge = {
                    'from_': a,
                    'to_': b,
                    'type_': undirectedEdgeType.id_
                }

                edgesToAdd.append(edge)

        graph.addEdges(edgesToAdd)

        return graph

    # Graph, Graph, Graph, Edge[]
    # Graph

    @staticmethod
    def remoralize(G1, G2, M1, directedEdgesToRemove=[]):
        toNodeNames = list(map(lambda e: e['to_'], directedEdgesToRemove))
        toNodes = GraphUtils.getNodesByName(toNodeNames, G1)
        PaToNodes = su.union(G1.parents(toNodes), toNodes, 'name')

        G1sub = GraphUtils.subgraph(G1, PaToNodes)
        G2sub = GraphUtils.subgraph(G2, PaToNodes)

        M1sub = GraphUtils.moralize(G1sub)
        M2sub = GraphUtils.moralize(G2sub)

        undirectedEdgesToRemove = su.difference(M1sub.edges, M2sub.edges)

        M2 = M1.copy()
        M2.deleteEdges(undirectedEdgesToRemove)

        return M2

    # Graph, Node[]
    # Graph

    @staticmethod
    def ancestral(G, V):
        V = ou.makeArray(V)

        An = GraphUtils.ancestors(V, G)
        AnG = GraphUtils.subgraph(G, V + An)

        return AnG

    # Node[][]

    @staticmethod
    def cCompDecomposition(G, sortNodesBeforeOrdering=False):
        assignment = {}
        number = 0
        decomp = []
        V = GraphUtils.topoSort(G, sortNodesBeforeOrdering)

        for node in V:
            if node['name'] not in assignment:
                # decomp[number] = []
                decomp.append([])
                assignment[node['name']] = number

                reachable = GraphUtils.reach(
                    node, G, bidirectedEdgeType, Direction.both)

                for rNode in reachable:
                    assignment[rNode['name']] = number
                    decomp[number].append(rNode)

                number = number + 1

        return decomp

    @staticmethod
    def hasCycles(G):
        return False

#     static hasCycles(G: Graph): boolean {
#         // don't test if undirected edge exist
#         for (let edge of G.edges) {
#             if (edge.type == undirectedEdgeType.id)
#                 return true;
#         }

#         // if we can find a topological order, then there is no cycle
#         try {
#             this.topoSort(G);
#             return false;
#         } catch (e) {
#             // how to figure out which edge caused cycle?
#             // it outputs node name (destination?)
#             // console.log(e)
#             return true;
#         }
#     }

    # str[][]

    @staticmethod
    def findCycles(G):
        return []

#     /**
#      * // https://stackoverflow.com/questions/30217313/how-to-improve-performance-of-finding-all-cycles-in-undirected-graphs
#      * // https://stackoverflow.com/questions/12367801/finding-all-cycles-in-undirected-graphs
#      * @param G
#      * @return list of cycles where each cycle lists a sequence of node names
#      */
#     static findCycles(G: Graph): string[][] {
#         let edges: string[][] = G.edges.map(e => [e.from, e.to]);
#         let cycles: string[][] = [];

#         for (let edge of edges) {
#             for (let j = 0; j < 2; j++) {
#                 this.findCycle(G, cycles, [edge[j]]);
#             }
#         }

#         return cycles;
#     }

    @staticmethod
    def __findCycle(G, cycles, path):
        return

#     private static findCycle(G: Graph, cycles: string[][], path: string[]) {
#         let edges: string[][] = G.edges.map((e) => [e.from, e.to]);

#         let startNode = path[0], nextNode: string;

#         // visit each edge and each node of each edge
#         for (let i = 0; i < edges.length; i++) {
#             let edge: string[] = edges[i];

#             for (var j = 0; j < 2; j++) {
#                 let node: string = edge[j];

#                 //  edge refers to our current node
#                 if (node === startNode) {
#                     nextNode = edge[(j + 1) % 2];

#                     //  neighbor node not on path yet
#                     if (!this.visited(nextNode, path))
#                         //  explore extended path
#                         this.findCycle(G, cycles, [nextNode].concat(path));

#                     //  cycle found
#                     else if ((path.length > 2) && (nextNode === path[path.length - 1])) {
#                         if (this.isNew(path, cycles))
#                             cycles.push(path);
#                     }
#                 }
#             }
#         }
#     }

    @staticmethod
    def __visited(node, path):
        return False

#     private static visited(node: string, path: string[]): boolean {
#         return (path.indexOf(node) !== -1);
#     }

    @staticmethod
    def __isNew(path, cycles):
        return True

#     private static isNew(path: string[], cycles: string[][]): boolean {
#         for (var i = 0; i < cycles.length; i++) {
#             if (this.isEqualPaths(path, cycles[i]))
#                 return false;
#         }

#         return true;
#     }

    @staticmethod
    def __isEqualPaths(p1, p2):
        return True

#     private static isEqualPaths(p1: string[], p2: string[]): boolean {
#         if (p1.length != p2.length) return false;

#         for (let i = 0; i < p1.length; i++) {
#             if (p1[i] != p2[i]) return false;
#         }

#         return true;
#     }

    @staticmethod
    def findIsomorphicGraphLabels(G1, G2):
        result = {'original': [], 'permuted': []}

        return result

#     static findIsomorphicGraphLabels(G1: Graph, G2: Graph): { original: string[], permuted: string[] } {
#         let degreesOriginal = this.getNodeDegrees(G1);
#         let degrees = this.getNodeDegrees(G2);

#         let G1NodeLabels: string[] = G1.nodes.map((n) => n.label);
#         let G2NodeLabels: string[] = G2.nodes.map((n) => n.label);

#         let permutations = this.findPermutations(G2NodeLabels);

#         for (let p of permutations) {
#             let found: boolean = true;

#             for (let i = 0; i < p.length; i++) {
#                 let originalLabel: string = G1NodeLabels[i];
#                 let permutedLabel: string = p[i];
#                 let degreesOriginalNode: { in: number, out: number } = degreesOriginal[originalLabel];
#                 let degreesPermutedNode: { in: number, out: number } = degrees[permutedLabel];

#                 if (degreesOriginalNode && degreesPermutedNode)
#                     if (degreesOriginalNode.in != degreesPermutedNode.in
#                         || degreesOriginalNode.out != degreesPermutedNode.out) {
#                         found = false;
#                         break;
#                     }
#             }

#             if (found) {
#                 return {
#                     original: G1NodeLabels,
#                     permuted: p
#                 };
#             }
#         }

#         return { original: [], permuted: [] };
#     }

    @staticmethod
    def getNodeDegrees(G):
        result = dict()

        return result

#     static getNodeDegrees(G: Graph): { [key: string]: { in: number, out: number } } {
#         if (!G || G.nodes.length == 0 || G.edges.length == 0) return {};

#         let degrees = Object.create(null);

#         for (let node of G.nodes) {
#             degrees[node.label] = {
#                 in: this.getIncoming(node, G).length,
#                 out: this.getOutgoing(node, G).length
#             };
#         }

#         return degrees;
#     }

    @staticmethod
    def findPermutations(list, result=[], usedChars=[]):
        return result

#     // reference: https://stackoverflow.com/questions/9960908/permutations-in-javascript
#     static findPermutations(list: any[], result: any[] = [], usedChars: any[] = []): any[] {
#         var i, ch;
#         for (i = 0; i < list.length; i++) {
#             ch = list.splice(i, 1)[0];
#             usedChars.push(ch);
#             if (list.length == 0) {
#                 result.push(usedChars.slice());
#             }
#             this.findPermutations(list, result, usedChars);
#             list.splice(i, 0, ch);
#             usedChars.pop();
#         }
#         return result;
#     }

#     static copyAnnotations(g: Graph, G: Graph) {
#         let annotationsToAdd = {};
#         let originalAnnotations: Annotation[] = G.metadata.annotations;

#         for (let annotId in originalAnnotations) {
#             let annot: Annotation = originalAnnotations[annotId];
#             let newAnnot: Annotation = {
#                 id: annotId,
#                 type: annot.type,
#                 metadata: ObjectUtils.clone(annot.metadata)
#             };

#             if (annot.type == labelAnnotationType.id) {
#                 let targetId: string = annot.metadata.targetId;

#                 if (!targetId) continue;

#                 let nodes: Node[] = G.nodes.filter((n) => n.id == targetId);

#                 if (nodes.length > 0) {
#                     let node: Node = nodes[0];
#                     let newNodes: Node[] = g.nodes.filter((n) => n.name == node.name);

#                     if (newNodes.length > 0) {
#                         let newNode: Node = newNodes[0];

#                         newNode.metadata.labelNodeId = newAnnot.id;

#                         annotationsToAdd[newAnnot.id] = newAnnot;
#                     }
#                 }

#                 let edges: Edge[] = G.edges.filter((e) => e.id == targetId);

#                 if (edges.length > 0) {
#                     let edge: Edge = edges[0];
#                     let newEdges: Edge[] = g.edges.filter((e) => e.label == edge.label);

#                     if (newEdges.length > 0) {
#                         let newEdge: Edge = newEdges[0];

#                         newEdge.metadata.labelNodeId = newAnnot.id;

#                         annotationsToAdd[newAnnot.id] = newAnnot;
#                     }
#                 }
#             } else
#                 annotationsToAdd[newAnnot.id] = newAnnot;
#         }

#         delete g.metadata.annotations;

#         g.metadata.annotations = annotationsToAdd;
#     }

#     static duplicateNodesEdges(nodes: Node[], edges: Edge[]): { nodes: Node[], edges: Edge[] } {
#         let newNodes: Node[] = [];
#         let newEdges: Edge[] = [];

#         for (let node of nodes) {
#             let newNode: Node = this.duplicateNode(node);

#             let newName: string = newNode.name + suffix;
#             let newLabel: string = newNode.name;

#             if (newNode.name == newNode.label)
#                 newLabel = newName;

#             newNode.name = newName;
#             newNode.label = newLabel;

#             if (newNode.metadata && newNode.metadata.x !== undefined && !isNaN(newNode.metadata.x))
#                 newNode.metadata.x = parseFloat(newNode.metadata.x) + 20;
#             if (newNode.metadata && newNode.metadata.y !== undefined && !isNaN(newNode.metadata.y))
#                 newNode.metadata.y = parseFloat(newNode.metadata.y) + 20;

#             if (newNode.metadata && newNode.metadata.labelNodeId)
#                 delete newNode.metadata.labelNodeId;

#             newNodes.push(newNode);
#         }

#         for (let edge of edges) {
#             let from: Node, to: Node;
#             let fromFilter: Node[] = newNodes.filter(n => { return n.name == edge.from + suffix });
#             if (fromFilter.length > 0)
#                 from = fromFilter[0];
#             let toFilter: Node[] = newNodes.filter(n => { return n.name == edge.to + suffix });
#             if (toFilter.length > 0)
#                 to = toFilter[0];

#             if (!from || !to) continue;
#             if (!from.id || !to.id) continue;

#             let newEdge: Edge = this.duplicateEdge(edge, from.id, to.id);
#             newEdge.id = this.generateEdgeId(edge, from.id, to.id);
#             newEdge.from = edge.from + suffix;
#             newEdge.to = edge.to + suffix;

#             // newEdges.push(this.duplicateEdge(edge, from.id, to.id, false));
#             newEdges.push(newEdge);
#         }

#         return { nodes: newNodes, edges: newEdges };
#     }

#     static duplicateNode(n: Node, simplified: boolean = false): Node {
#         let node: Node = Object.create(null);
#         node.id = !simplified ? n.id : this.generateNodeId();
#         // node.id = cloned ? n.id : this.generateNodeId();
#         node.name = n.name;
#         node.label = n.label ? n.label : n.name;
#         node.type = n.type;
#         if (n.metadata)
#             node.metadata = ObjectUtils.clone(n.metadata);
#         else
#             node.metadata = Object.create(null);
#         if (node.metadata.forceLabelWithoutSubscript)
#             node.label = 'S';
#         if (node.id != node.metadata.id)
#             node.metadata.id = node.id;

#         return node;
#     }

#     static duplicateEdge(e: Edge, fromId: string, toId: string, simplified: boolean = false): Edge {
#         let edge: Edge = Object.create(null);
#         edge.from = e.from;
#         edge.to = e.to;
#         edge.type = e.type ? e.type : directedEdgeType.id;
#         edge.label = e.label;
#         // edge.id = !simplified ? e.id : this.generateEdgeId(edge, e.from, e.to);
#         edge.id = !simplified ? e.id : this.generateEdgeId(edge, fromId, toId);
#         // edge.id = cloned ? e.id : this.generateEdgeId(edge, fromId, toId);

#         // if (cloned)
#         //     edge.id = e.id;
#         // else
#         //     edge.id = simplified ? this.generateEdgeId(edge, fromId, toId) : e.id;

#         if (e.metadata)
#             edge.metadata = ObjectUtils.clone(e.metadata);
#         else
#             edge.metadata = Object.create(null);
#         if (edge.id != edge.metadata.id)
#             edge.metadata.id = edge.id;
#         edge.metadata.from = fromId;
#         edge.metadata.to = toId;
#         // edge.metadata.from = !simplified ? fromId : e.from;
#         // edge.metadata.to = !simplified ? toId : e.to;
#         // edge.metadata.from = cloned ? e.from : fromId;
#         // edge.metadata.to = cloned ? e.to : toId;

#         return edge;
#     }

#     static duplicateTask(task: Task): Task {
#         let newTask = new BasicTask();
#         newTask.type = task.type;
#         newTask.sets = ObjectUtils.clone(task.sets);
#         newTask.definitions = ObjectUtils.clone(task.definitions);
#         newTask.collections = ObjectUtils.clone(task.collections);

#         return newTask;
#     }

    # str, Graph
    # Node

    @staticmethod
    def getNodeByName(name, G):
        if not name or not G or not G.nodes:
            return None

        return next((n for n in G.nodes if GraphUtils.correctNodeName(n['name']) == GraphUtils.correctNodeName(name)), None)

    # str | str[], Graph
    # Node

    @staticmethod
    def getNodesByName(names, G):
        if not names or not G:
            return []

        names = ou.makeArray(names)
        dNames = dict()

        for name in names:
            if isinstance(name, str):
                dNames[GraphUtils.correctNodeName(name)] = False
            else:
                v = name
                dNames[GraphUtils.correctNodeName(v['name'])] = False

        for node in G.nodes:
            if GraphUtils.correctNodeName(node['name']) in dNames:
                dNames[GraphUtils.correctNodeName(node['name'])] = node

        nodes = []

        for name in names:
            actualName = ''

            if isinstance(name, str):
                actualName = GraphUtils.correctNodeName(name)
            else:
                actualName = GraphUtils.correctNodeName(node['name'])

            if actualName in dNames and dNames[actualName] is not False:
                nodes.append(dNames[actualName])

        return nodes

    # str, str, Graph, EdgeType
    # Edge

    @staticmethod
    def getEdgeByName(from_, to_, G, type_=None):
        if not G or not G.edges:
            return None

        fromNode = GraphUtils.getNodeByName(from_, G)
        toNode = GraphUtils.getNodeByName(to_, G)

        if not fromNode or not toNode:
            return None

        if not type_:
            return next((e for e in G.edges if e['from_'] == fromNode['name'] and e['to_'] == toNode['name']), None)
        else:
            return next((e for e in G.edges if e['from_'] == fromNode['name'] and e['to_'] == toNode['name'] and e['type_'] == type_.id_), None)

    @staticmethod
    def hasEdge(from_, to_, G, type=None):
        return G.nx.has_edge(from_, to_)

    # G, boolean
    # Node[]

    @staticmethod
    def topoSort(G, sort_=False):
        nodeNames = list(map(lambda n: n['name'], G.nodes))

        # create dict of dependencies to run toposort
        # format: {to: {from1, from2, ...}}
        dep = dict()

        for edge in G.edges:
            # directed edges only
            if edge['type_'] != directedEdgeType.id_:
                continue

            if edge['to_'] not in dep:
                dep[edge['to_']] = set()

            dep[edge['to_']].add(edge['from_'])

        nodeNames = toposort_flatten(dep, sort_)

        V = GraphUtils.getNodesByName(nodeNames, G)

        return su.union(V, G.nodes, 'name')


#     static minimizeSets(sets: Node[][]): Node[][] {
#         if (!sets) return [];

#         let newSets = sets.slice();

#         let minSize: number = 1e8;

#         for (let nodes of newSets) {
#             if (nodes && nodes.length < minSize)
#                 minSize = nodes.length;
#         }

#         newSets = newSets.filter(nodes => !nodes || (nodes && nodes.length <= minSize));

#         return newSets;
#     }

#     static sortSets(sets: Node[][]): Node[][] {
#         if (!sets) return [];

#         let newSets = sets.slice();

#         // sort results
#         // sort names within each set of nodes
#         for (let nodes of newSets) {
#             if (nodes)
#                 nodes = nodes.sort(sortByName);
#         }

#         // sort sets of nodes
#         newSets.sort((sep1, sep2) => {
#             let shorterLength: number = sep1.length > sep2.length ? sep2.length : sep1.length;

#             for (let i = 0; i < shorterLength; i++) {
#                 let a: Node = sep1[i];
#                 let b: Node = sep2[i];

#                 if (a.name < b.name) return -1;
#                 else if (a.name > b.name) return 1;
#             }

#             return 0;
#         });

#         return newSets;
#     }

    @staticmethod
    def nodesToVariables(nodes):
        nodes = ou.makeArray(nodes)

        variables = []

        for node in nodes:
            if 'label' in node:
                variables.append(
                    {'name': node['name'], 'label': node['label']})
            else:
                variables.append({'name': node['name']})

        return variables

    @staticmethod
    def toString(G):
        return ''

#     static toString(G: Graph): string {
#         return G.nodes.map((node) => { return node.toString(); }).join(", ") + "; " + G.edges.map((edge) => { return edge.toString(); }).join(", ");
#     }

    @staticmethod
    def correctNodeName(name):
        if not name:
            return ''

        return name.strip().upper()

    # Node | Node[]
    # Dict[key, boolean]
    @staticmethod
    def nodeToNameMap(nodes):
        nodes = ou.makeArray(nodes)

        names = dict()

        for node in nodes:
            names[node['name']] = True

        return names

    @staticmethod
    def nodeToList(nodes, attribute='name'):
        nodes = ou.makeArray(nodes)

        attrs = []

        for node in nodes:
            attrs.append(node[attribute])

        return attrs


#     static generateNodeId() {
#         return UUIDGenerator.generateRandomID(32);
#     }

#     static generateEdgeId(edge: Edge, fromId: string, toId: string): string {
#         if (!fromId || !toId) return '';
#         return fromId.toUpperCase() + edgeTypeMap.get(edge.type).shortId + toId.toUpperCase();
#     }

#     static parseEdgeId(edge: Edge): { from: string, to: string } {
#         if (!edge || !edge.id || !edge.type) return null;

#         let shortId: string = edgeTypeMap.get(edge.type).shortId;
#         let tokens: string[] = edge.id.split(shortId);

#         if (tokens.length != 2) return null;

#         return {
#             from: tokens[0],
#             to: tokens[1]
#         };
#     }

    @staticmethod
    def mapNodeNames(nodes):
        nodes = ou.makeArray(nodes)

        namesMap = dict()

        for node in nodes:
            namesMap[node['name']] = True

        return namesMap

#     private static mapNodeNames(nodes: Node[] | Node): { [name: string]: boolean } {
#         nodes = ObjectUtils.ou.makeArray(nodes);
#         return nodes.reduce((map, n) => { map[n.name] = true; return map; }, Object.create(null));
#     }

# }