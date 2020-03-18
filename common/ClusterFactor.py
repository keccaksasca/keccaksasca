from Node import Node
from Edge import Edge
from abc import ABC, abstractmethod
import numpy as np
import settings
from typing import List

##################
#                #
#  (A)           #
#     \          #
#     [?]---(C)  #
#    /           #
#  (B)           #
#                #
##################

# Abstract Base Class (ABC) Factor
class ClusterFactor(ABC):
    edges: List[Edge]
    name: str

    def __init__(self,name):
        #   [A,B,C,Prob]
        self.edges = []
        self.name = name


    def __repr__(self):
        return f"{self.__class__.__name__} {self.name}"

    def initMessages(self):
        for edge in self.edges:
            numbits = edge.cardinality
            vals = 2**numbits
            edge.m2n = np.full(shape=vals, fill_value=1.0/vals, dtype=settings.NUMPY_DATATYPE)

        self.IODirections = np.array(self.IODirections)

    # generic function, just puts all messages in a list
    def gatherIncoming(self):
        return [edge.m2f for edge in self.edges]

    @abstractmethod
    def f2n(self):
        pass
