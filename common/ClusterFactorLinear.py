from Node import Node
from Factor import Factor
import numpy as np
import settings
import ffht

def wht(msgin):
    for i in range(msgin.shape[0]):
        ffht.fht(msgin[i, :])
    return msgin

# linear factor --> everything that can be written using just XORs (such as theta, parity, etc)
# this variant --> all messages are aligned! --> can be used for computing parity and for theta, but not for theta effect
class ClusterFactorLinear(Factor):
    def __init__(self, name):
        self.numbits = settings.CLUSTERSIZE
        self.numvalues = 2**self.numbits
        super().__init__(name)

    # packed in own function to keep core the same
    def gatherIncoming(self):
        msgin = np.zeros(shape=(len(self.edges), self.numvalues), dtype=settings.NUMPY_DATATYPE)
        for (idx, edge) in enumerate(self.edges):
            msgin[idx, :] = edge.m2f

        return msgin

    # packed in own function to keep core the same
    def spreadOutgoing(self, msgout):
        for (edgeidx, edge) in enumerate(self.edges):
            edge.m2n = msgout[edgeidx, :]

    # efficient factor2node using the Walsh-Hadamard Transfrom
    def f2n(self):
        msgin = self.gatherIncoming()

        l = msgin.shape[0]

        # perform the transform on all inputs
        msgw = wht(msgin)

        # do the product
        msgoutw = np.zeros(shape=(l, self.numvalues), dtype=settings.NUMPY_DATATYPE)
        idxall = np.full(l, True)
        for targetIdx in range(l):
            curridx = idxall.copy()
            curridx[targetIdx] = False

            m = msgw[curridx, :]
            msgoutw[targetIdx, :] = np.prod(m, axis=0)

        # transform back
        # msgout = (settings.WMAT @ msgoutw.T).T
        msgout = wht(msgoutw)/self.numvalues #wht is an involution, up to the scaling

        self.spreadOutgoing(msgout)
