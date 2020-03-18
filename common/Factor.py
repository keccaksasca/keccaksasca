from abc import ABC, abstractmethod
import numpy as np
import settings
from typing import List
from Edge import Edge

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

class Factor(ABC):
    edges: List[Edge]
    name: str

    def __init__(self,name):
        #   [A,B,C,Prob]
        self.edges = []
        self.name = name
        self.isLeaf = False

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name}"

    def initMessages(self):
        for edge in self.edges:
            numbits = edge.cardinality
            vals = 2**numbits
            edge.m2n = np.full(shape=vals, fill_value=1.0/vals, dtype=settings.NUMPY_DATATYPE)

    def gatherIncoming(self):
        l = len(self.edges)
        msgin = np.zeros(shape=(l, 2), dtype=settings.NUMPY_DATATYPE)
        for (idx, edge) in enumerate(self.edges):
            msgin[idx, :] = edge.m2f
        return msgin

    @abstractmethod
    def f2n(self):
        pass
