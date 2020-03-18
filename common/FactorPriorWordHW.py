from Node import Node
from Factor import Factor
from FactorGeneric import FactorGeneric
from TraitPrior import TraitPrior
import utils
import numpy as np
import settings

###############
#             #
#     [F]     #
#   /    \    #
#  (A)...(A)  #
#             #
###############

def pbpdf(probin):
    c = FactorPriorWordHW.c
    p = np.outer(probin, c-1)
    p_log = np.sum(np.log1p(p), axis=0)
    s = np.exp(p_log)
    s = np.abs(np.real(np.fft.fft(s)))
    pdf = s/(len(probin)+1)

    return pdf

def pbpdf_c(l):
    l = l-1
    return np.exp(2*1j*np.pi*np.arange(0, l+1)/(l+1))

class FactorPriorWordHW(Factor, TraitPrior):

    c = None

    def __init__(self,name,wordsize,dist):
        assert(len(dist)==(wordsize+1))

        self.dist = dist #for debugging
        self.wordsize = wordsize

        super().__init__(name)


    def f2n(self):
        l = len(self.edges)

        if(FactorPriorWordHW.c is None):
            FactorPriorWordHW.c = pbpdf_c(l)

        msgin = self.gatherIncoming()

        # now iterate over all the target nodes
        idxall = np.full(l, True)
        for (targetIdx,edge) in enumerate(self.edges):
            curridx = idxall.copy()
            curridx[targetIdx] = False

            currmsg = msgin[curridx, 1]
            currpdf = pbpdf(currmsg)

            p0 = np.dot(self.dist[:-1], currpdf)
            p1 = np.dot(self.dist[1:], currpdf)

            edge.m2n = np.array([p0, p1])

