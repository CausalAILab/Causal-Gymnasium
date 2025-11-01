import networkx as nx
from typing import List, Dict, Any

from .graph_defs import basicNodeType, directedEdgeType, bidirectedEdgeType, EdgeType, edgeTypeMap

from .graph_utils import GraphUtils as gu
from .set_utils import SetUtils as su
from .object_utils import ObjectUtils as ou


class Graph():
    
    name = str
    # task = Task
    metadata = Dict[str, Any]

    def __init__(self, graph = None, nodes = [], edges = [], task = None, metadata = dict()):
        if graph is not None:
            if isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph):
                self.nx = graph
            elif isinstance(graph, Graph):
                self.nx = graph.nx
        else:
            self.nx = nx.DiGraph()
            self.addNodes(nodes)
            self.addEdges(edges)

        # if task is None:
        #     task = BasicTask()

        # self.task = task
        self.metadata = metadata


    @property
    def nodes(self):
        nodes = self.nx.nodes(data = True) 
        ns = []

        # convert tuple to list of dictionaries
        for node in nodes:
            n = {
                'name': node[0],
                'label': node[1]['label'] if 'label' in node[1] else node[0],
                'type_': node[1]['type_'] if 'type_' in node[1] else basicNodeType.id_,
                'metadata': node[1]['metadata'] if 'metadata' in node[1] else {}
            }

            ns.append(n)

        return ns

        # return self.nx.nodes(data = True)


    @nodes.setter
    def nodes(self, nodes):
        self.nx.clear()
        self.addNodes(nodes)


    @property
    def edges(self):
        edges = self.nx.edges(data = True)
        es = []

        for edge in edges:
            e = {
                'from_': edge[0],
                'to_': edge[1],
                'label': edge[2]['label'] if 'label' in edge[2] else None,
                'type_': edge[2]['type_'] if 'type_' in edge[2] else directedEdgeType.id_,
                'metadata': edge[2]['metadata'] if 'metadata' in edge[2] else {}
            }

            es.append(e)

        return es

        # return self.nx.edges(data = True)

    @edges.setter
    def edges(self, edges):
        self.deleteEdges(self.edges)
        self.addEdges(edges)


    def copy(self):
        return Graph(self.nx.copy())


    def toUndirected(self):
        # return self.nx.to_undirected()
        self.nx = nx.Graph(self.nx)

        # change all edge types


    def addNodes(self, nodes):
        nodes = ou.makeArray(nodes)
        
        if not nodes or len(nodes) == 0:
            return
        
        for node in nodes:
            # should we throw error if name is not specified?
            # duplicate name/label?

            # if not hasattr(node, 'name'):
            if 'name' not in node:
                continue

            # if not hasattr(node, 'label'):
            if 'label' not in node:
                node['label'] = node['name']

            # if not hasattr(node, 'type_'):
            if 'type_' not in node:
                node['type_'] = basicNodeType.id_

            # if not hasattr(node, 'metadata'):
            if 'metadata' not in node:
                node['metadata'] = dict()
            
            self.__addNode(node)


    def deleteNodes(self, nodes):
        nodes = ou.makeArray(nodes)

        if not nodes or len(nodes) == 0:
            return
        
        self.deleteEdges(gu.getIncoming(nodes, self))
        self.deleteEdges(gu.getOutgoing(nodes, self))
        
        # can we delete all at once instead?
        for node in nodes:
            self.__deleteNode(node)


    def addEdges(self, edges):
        edges = ou.makeArray(edges)

        if not edges or len(edges) == 0:
            return

        for edge in edges:
            # if not hasattr(edge, 'from_') or not hasattr(edge, 'to_'):
            if 'from_' not in edge or 'to_' not in edge:
                continue

            # if not hasattr(edge, 'label'):
            if 'label' not in edge:
                edge['label'] = None

            # if not hasattr(edge, 'type_'):
            if 'type_' not in edge:
                edge['type_'] = directedEdgeType.id_

            # if not hasattr(edge, 'metadata'):
            if 'metadata' not in edge:
                edge['metadata'] = dict()
            
            # nx has a 'bug': if X -- Y exists, adding X -> Y will overwrite X -- Y, or vice versa
            if self.nx.has_edge(edge['from_'], edge['to_']):
                existingEdge = None

                for e in self.edges:
                    if e['from_'] == edge['from_'] and e['to_'] == edge['to_']:
                        existingEdge = e
                        break
                
                # if adding X -> Y, remove X -- Y and add Y -- X first
                if edge['type_'] == directedEdgeType.id_:
                    if existingEdge['type_'] == bidirectedEdgeType.id_:
                        self.deleteEdges(existingEdge)
                        
                        temp = existingEdge['from_']
                        existingEdge['from_'] = existingEdge['to_']
                        existingEdge['to_'] = temp
                        
                        self.addEdges(existingEdge)
                    else:
                        edgeType = directedEdgeType

                        for id_ in edgeTypeMap:
                            if edge['type_'].upper() == edgeTypeMap[id_].id_.upper():
                                edgeType = edgeTypeMap[id_]
                                break

                        raise Exception('The edge ' + edge['from_'] + ' ' + edgeType.shortId + ' ' + edge['to_'] + ' already exists.')

                # if adding X -- Y, flip from/to and add Y -- X instead
                elif edge['type_'] == bidirectedEdgeType.id_:
                    if existingEdge['type_'] == directedEdgeType.id_:
                        temp = edge['from_']
                        edge['from_'] = edge['to_']
                        edge['to_'] = temp
                    else:
                        edgeType = directedEdgeType

                        for id_ in edgeTypeMap:
                            if edge['type_'].upper() == edgeTypeMap[id_].id_.upper():
                                edgeType = edgeTypeMap[id_]
                                break

                        raise Exception('The edge ' + edge['from_'] + ' ' + edgeType.shortId + ' ' + edge['to_'] + ' already exists.')

            # test if the new edge generates a cycle
            self.__addEdge(edge)


    def deleteEdges(self, edges):
        edges = ou.makeArray(edges)

        if not edges or len(edges) == 0:
            return

        # can we delete all at once instead?
        for edge in edges:
            self.__deleteEdge(edge)


    # Node[]
    # Node[]
    def parents(self, nodes):
        nodes = ou.makeArray(nodes)
        
        if not nodes or len(nodes) == 0:
            return []

        names = dict()
        parents = dict()
        
        for node in nodes:
            names[node['name']] = True
        
        for edge in self.edges:
            if edge['type_'] == directedEdgeType.id_ and edge['to_'] in names:
                parents[edge['from_']] = True
        
        p = list(filter(lambda n: n['name'] in parents, self.nodes))
        
        return p


    def ancestors(self, nodes):
        nodes = ou.makeArray(nodes)

        if not nodes or len(nodes) == 0:
            return []

        AnNames = []

        for node in nodes:
            AnNames.extend(list(nx.ancestors(self.nx, node['name'])))
        
        AnNames = su.unique(AnNames)

        An = gu.getNodesByName(AnNames, self)
        An = su.difference(An, nodes, 'name')
        
        return An


    def descendants(self, nodes):
        nodes = ou.makeArray(nodes)
        
        if not nodes or len(nodes) == 0:
            return []
        
        DeNames = []
        
        # bug: nx treats bidirected as directed
        for node in nodes:
            DeNames.extend(list(nx.descendants(self.nx, node['name'])))
        
        DeNames = su.unique(DeNames)
        De = gu.getNodesByName(DeNames, self)
        De = su.difference(De, nodes, 'name')
        
        return De


    def children(self, nodes):
        nodes = ou.makeArray(nodes)

        names = dict()
        children = dict()

        for node in nodes:
            names[node['name']] = True
        
        for edge in self.edges:
            if edge['type_'] == directedEdgeType.id_ and edge['from_'] in names:
                children[edge['to_']] = True

        c = list(filter(lambda n: n['name'] in children, self.nodes))
        
        return c


    # Node | Node[], EdgeType | str
    # Node[]
    def neighbors(self, nodes, edgeType = None):
        nodes = ou.makeArray(nodes)
        
        if not nodes or len(nodes) == 0:
            return []
        
        if isinstance(edgeType, EdgeType):
            edgeType = edgeType.id_
        
        names = dict()
        neighbors = dict()
        
        for node in nodes:
            names[node['name']] = True

        edges = list(filter(lambda e: e['type_'] == edgeType if edgeType else True, self.edges))
        
        for edge in edges:
            if edge['to_'] in names:
                neighbors[edge['from_']] = True
            if edge['from_'] in names:
                neighbors[edge['to_']] = True
        
        n = list(filter(lambda n: n['name'] in neighbors and n['name'] not in names, self.nodes))
        
        return n


    def connectedComponents(self):
        # node names
        # convert to nodes
        CC = list(nx.connected_components(self.nx))
        ccsNodes = []

        # cc is a dict
        for cc in CC:
            nodeNames = []

            for nodeName in cc:
                nodeNames.append(nodeName)
            
            ccNodes = gu.getNodesByName(nodeNames, self)
            
            ccsNodes.append(ccNodes)
        
        return ccsNodes


    def print(self):
        if len(self.nodes) > 0:
            print('<Nodes (' + str(len(self.nodes)) + ')>')
        else:
            print('<Nodes (none)>')

        nodeNames = list(map(lambda n: n['name'], self.nodes))

        print('\n'.join(nodeNames))

        if len(self.edges) > 0:
            print('\n<Edges (' + str(len(self.edges)) + ')>')
        else:
            print('\n<Edges (none)>')

        edgeList = list(map(lambda e: e['from_'] + ' ' + edgeTypeMap[e['type_']].shortId  + ' ' + e['to_'], self.edges))

        print('\n'.join(edgeList))


    def __addNode(self, node):
        self.nx.add_node(node['name'], label = node['label'], type_ = node['type_'], metadata = node['metadata'])

    def __deleteNode(self, node):
        self.nx.remove_node(node['name'])

    def __addEdge(self, edge):
        self.nx.add_edge(edge['from_'], edge['to_'], label = edge['label'], type_ = edge['type_'], metadata = edge['metadata'])

    def __deleteEdge(self, edge):
        self.nx.remove_edge(edge['from_'], edge['to_'])