class NodeType():

    id_: str
    name: str
    shortId: str

    def __init__(self, id_, name, shortId):
        self.id_ = id_
        self.name = name
        self.shortId = shortId

class EdgeType():

    id_: str
    name: str
    shortId: str

    def __init__(self, id_, name, shortId):
        self.id_ = id_
        self.name = name
        self.shortId = shortId

def isNodeType(nodeType):
    return isinstance(nodeType, NodeType)

def isEdgeType(edgeType):
    return isinstance(edgeType, EdgeType)

# how to change to enum?
basicNodeType = NodeType('basic', 'Basic', '')
latentNodeType = NodeType('latent', 'Latent', 'L')

directedEdgeType = EdgeType('directed', 'Directed', '->')
bidirectedEdgeType = EdgeType('bidirected', 'Bidirected', '--')
undirectedEdgeType = EdgeType('undirected', 'Undirected', '-')

nodeTypeMap = dict()
nodeTypeMap[basicNodeType.id_] = basicNodeType
nodeTypeMap[latentNodeType.id_] = latentNodeType

edgeTypeMap = dict()
edgeTypeMap[directedEdgeType.id_] = directedEdgeType
edgeTypeMap[bidirectedEdgeType.id_] = bidirectedEdgeType
edgeTypeMap[undirectedEdgeType.id_] = undirectedEdgeType