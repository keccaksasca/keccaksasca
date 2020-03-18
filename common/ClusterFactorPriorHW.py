import numpy as np
from Factor import Factor
from TraitPrior import TraitPrior
import settings
import functools
from itertools import compress
from utils import popcount

class ClusterFactorPriorHW(Factor, TraitPrior):

    def __init__(self,name,wordsize,dist):
        assert(len(dist)==(wordsize+1))
        self.wordsize = wordsize
        self.numvals = 2**wordsize
        self.dist = dist

        super().__init__(name)

    def initMessages(self):
        totcard = sum([x.cardinality for x in self.edges])
        assert totcard == self.wordsize

        if len(self.edges) == 1: #if leaf --> just broadcast entire distribution on edge. Done in init, not needed later
            self.isLeaf = True

            edge = self.edges[0]
            assert(self.wordsize == edge.cardinality)

            msgout = np.zeros(shape=self.numvals, dtype=settings.NUMPY_DATATYPE)
            # for i in range(self.numvals):
            #     msgout[i] = self.dist[popcount(i)]
            hws = settings.HWTABLE[:2**self.wordsize]
            for i in range(self.wordsize+1):
                idx = (hws == i)
                msgout[idx] = self.dist[i]

            msgout = msgout / np.sum(msgout)

            edge.m2n = msgout

        else:
            super().initMessages()
            self.nodeTable = np.zeros(shape=totcard, dtype=int)

            nodeidx = 0
            for (edgeidx, edge) in enumerate(self.edges):
                vals = 2**edge.cardinality
                factorexpand = np.zeros(shape=vals, dtype=int)
                for i in range(vals):
                    factorexpand[i] = popcount(i)

                edge.factorexpand = factorexpand
                for i in range(edge.cardinality):
                    self.nodeTable[nodeidx] = edgeidx
                    nodeidx += 1

    def f2n(self):
        if self.isLeaf:
            pass #no need to do anything (message is already out)

        else:
            #gather messages, for each message compute hw-distribution
            hwdists = []
            for edge in self.edges:
                hwdist = np.zeros(shape=edge.cardinality+1, dtype=settings.NUMPY_DATATYPE)
                for (val, p) in enumerate(edge.m2f):
                    hwdist[edge.factorexpand[val]] += p
                hwdists += [hwdist]

            idx = [True]*len(self.edges)
            for (edgeidx, edge) in enumerate(self.edges):
                selfbits = edge.cardinality
                otherbits = self.wordsize - selfbits
                idx[edgeidx] = False
                otherhw = compress(hwdists, idx)

                # add two hamming-weight probability vectors --> equivalent to convolution
                otherpdf = functools.reduce(lambda x, y: np.convolve(x, y), otherhw)

                # now we have to do another convolution-type thingy (could be done more efficiently using fast convolution, but whatever)
                targethwprob = np.zeros(shape=selfbits+1, dtype=settings.NUMPY_DATATYPE)
                for targethw in range(selfbits+1):
                    for otherhw in range(otherbits+1):
                        factorhw = targethw+otherhw
                        targethwprob[targethw] += self.dist[factorhw]*otherpdf[otherhw]

                #finally, need to expand stuff
                edge.m2n = targethwprob[edge.factorexpand]

                idx[edgeidx] = True
