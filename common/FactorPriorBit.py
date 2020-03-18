#from Node import Node
from Factor import Factor
from TraitPrior import TraitPrior
import numpy as np
import settings

#########
#       #
#  [F]  #
#   |   #
#  (A)  #
#       #
#########

class FactorPriorBit(Factor, TraitPrior):
    def __init__(self,name,dist):
        assert(len(dist)==2)
        self.dist = np.array(dist,dtype=settings.NUMPY_DATATYPE)
        super().__init__(name)

    def f2n(self):
        edge = self.edges[0]
        edge.m2n = self.dist
