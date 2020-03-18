import numpy as np
#important: do not do "from x import x" for Factor and Node --> cyclic inclusion...
import Factor
import Node
import settings
from typing import List
from BPEnums import IODirection

class Edge(object):
    node: Node
    factor: Factor
    cardinality: int
    nodeIndizes: List[int]
    # IN:    node is input of factor
    # OUT:   node is input of factor
    # PRIOR: priorfactor, no direction
    direction: IODirection

    m2f: np.array
    m2n: np.array

    def __init__(self, Node, Factor, direction, nodeIndizes = None):
        self.node = Node
        self.factor = Factor

        if nodeIndizes is None:
            self.nodeIndizes = np.arange(settings.CLUSTERSIZE)
        else:
            self.nodeIndizes = nodeIndizes

        self.cardinality = len(self.nodeIndizes)
        self.direction = direction

    def __repr__(self):
        return f"{self.node.name} <-> {self.factor.name}"

