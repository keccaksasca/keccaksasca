from Node import Node
from Factor import Factor
import numpy as np
import settings

# linear factor --> everything that can be written using just XORs (such as theta, parity, etc)
class FactorLinear(Factor):
    def __init__(self, name):
        super().__init__(name)

    # efficient factor2node using the piling-up lemma
    def f2n(self):
        l = len(self.edges)

        msgin = self.gatherIncoming()
        msgin = msgin[:, 0] #we only care about the probability that the bit is 0

        bias = msgin - 0.5

        if self.name in settings.watchfactors:
            t = 0 # debug line

        idxall = np.full(l, True)
        n = l-1
        for (targetIdx, edge) in enumerate(self.edges):
            curridx = idxall.copy()
            curridx[targetIdx] = False

            p0 = 0.5 + (2**(n-1))*np.prod(bias[curridx])
            p1 = 1 - p0
            dist = np.maximum(np.array([p0, p1]), 0)
            edge.m2n = np.array(dist,dtype=settings.NUMPY_DATATYPE)



