import numpy as np
from Factor import Factor
from TraitPrior import TraitPrior
import settings

class ClusterFactorPriorHard(Factor, TraitPrior):

    def __init__(self,name,wordsize,value):
        
        self.wordsize = wordsize
        self.numvals = 2**wordsize
        self.value = value
        
        self.dist = np.zeros(shape=2**wordsize, dtype=settings.NUMPY_DATATYPE)
        self.dist[value] = 1

        self.isLeaf = True

        super().__init__(name)
    
    def initMessages(self):
        assert len(self.edges) == 1
        self.edges[0].m2n = self.dist

    def f2n(self):
        pass
