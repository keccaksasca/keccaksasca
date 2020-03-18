# from settings import NUMPY_DATATYPE, DAMP
import settings
import numpy as np
from utils import nName, intToBits
from copy import deepcopy

from TraitPrior import TraitPrior
from BPEnums import *

from typing import List
import Edge
import ClusterFactor
from scipy.stats import entropy
from utils import bitMarginalize


class ClusterNode(object):
    edges: List[Edge.Edge]
    entropy: float
    numbits: int
    numvalues: int
    change: float
    x: int
    z: int
    name: str
    value: int

    IODirections: List[IODirection]
    inFactorIdx: List[ClusterFactor.ClusterFactor]
    outFactorIdx: List[ClusterFactor.ClusterFactor]
    priorFactorIdx: List[ClusterFactor.ClusterFactor]

    def __init__(self, x, y, z, t, numbits):
        self.edges = []
        self.entropy = numbits
        self.numbits = numbits
        self.numvalues = 2**numbits
        self.prevDist = np.full(shape=self.numvalues, fill_value=1/self.numvalues, dtype=settings.NUMPY_DATATYPE)
        self.finalDist = np.full(shape=self.numvalues, fill_value=1/self.numvalues, dtype=settings.NUMPY_DATATYPE)
        self.change = 0
        # coordinates stored separately, for easier filtering etc.
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.name = nName(x, y, z, t)

        self.prevOut = None

        self.value = 0

        self.bits = np.full(shape=self.numbits, fill_value=0, dtype=np.uint8)
        self.bitsDist = np.full(shape=(self.numbits, 2), fill_value=0.5, dtype=settings.NUMPY_DATATYPE)
        self.bitsEntropy = np.ones(shape=self.numbits, dtype=settings.NUMPY_DATATYPE)

        self.IODirections = []
        self.inFactorIdx = []
        self.outFactorIdx = []
        self.priorFactorIdx = []

        self.trueValue = 0
    def __repr__(self):
        return f"Node {self.name}: ({self.value}, {self.finalDist[self.value]:.2f})"

    def initMessages(self):
        for edge in self.edges:
            numbits = edge.cardinality
            vals = 2**numbits
            edge.m2f = np.full(shape=vals, fill_value=1.0/vals, dtype=settings.NUMPY_DATATYPE)

            if edge.cardinality < self.numbits:
                nrows = 2**edge.cardinality
                ncols = 2**(self.numbits  - edge.cardinality)

                bitidx = edge.nodeIndizes
                bm = 0 #bitmask
                for i in range(numbits):
                    bm |= (1 << bitidx[i])

                t1 = np.bitwise_and(np.arange(2**self.numbits, dtype=int), bm) #the indexed bits
                t0 = np.bitwise_xor(np.arange(2**self.numbits, dtype=int), t1) #all other bits

                tx = t1[t0 == 0]
                tab = np.zeros(shape=2**self.numbits, dtype=int)
                tab[tx] = np.arange(2**numbits, dtype=int)
                edge.nodeExpand = tab[t1]

                tx = t0[t1 == 0]
                tab = np.zeros(shape=2**self.numbits, dtype=int)
                tab[tx] = np.arange(2**(self.numbits - numbits), dtype=int)
                tab = tab[t0]

                edge.nodeContract = np.zeros(shape=(nrows, ncols), dtype=int)
                edge.nodeContract[edge.nodeExpand, tab] = np.arange(2**self.numbits, dtype=int)

        self.IODirections = np.array([edge.direction.value for edge in self.edges])
        self.inFactorIdx = np.flatnonzero(self.IODirections == IODirection.In.value)
        self.outFactorIdx = np.flatnonzero(self.IODirections == IODirection.Out.value)
        noleafList = [not edge.factor.isLeaf for edge in self.edges]
        self.priorFactorIdx = np.flatnonzero(np.logical_and(self.IODirections == IODirection.Prior.value, noleafList))

        self.prevIn = None
        self.prevOut = None


    def gatherIncoming(self):
        msgin = np.zeros(shape=(len(self.edges), self.numvalues), dtype=settings.NUMPY_DATATYPE)
        for (idx, edge) in enumerate(self.edges):
            if edge.cardinality == self.numbits:
                msgin[idx, :] = edge.m2n
            else: # have to spread out stuff
                msgin[idx, :] = edge.m2n[edge.nodeExpand]

        msgin = np.maximum(msgin, 0) #just for the case some factors act up
        return msgin

    def spreadOutgoing(self, msgout, idx):
        # for (edgeidx, edge) in enumerate(self.edges):
        for edgeidx in idx:
            edge = self.edges[edgeidx]
            if edge.cardinality == self.numbits:
                edge.m2f = msgout[edgeidx, :]
            else: #have to sum out other variables
                m = msgout[edgeidx, :]
                m = m[edge.nodeContract]
                m = np.sum(m, axis=1)
                edge.m2f = m

    def n2f(self, target=None):
        d = settings.DAMP
        l = len(self.edges)

        msgin = self.gatherIncoming()
        # damping at input
        if self.prevIn is not None:
            msgin = d*msgin + (1-d)*self.prevIn

        msgout = np.zeros(shape=(l, self.numvalues), dtype=settings.NUMPY_DATATYPE)

        if self.name in settings.watchnodes:
            print(f"{self.name}: {self.entropy}")

        # if target is None:
        targets = range(l)
        if(target == IODirection.In):
            targets = self.inFactorIdx
        elif(target == IODirection.Out):
            targets = self.outFactorIdx
        elif(target == IODirection.Prior):
            targets = self.priorFactorIdx

        curridx = np.full(l, True)
        #now perform multiplications for each target factor
        for targetIdx in targets:
            curridx[targetIdx] = False

            #product
            p = np.prod(msgin[curridx, :], 0)

            s = np.sum(p)
            if s == 0:
                print(f"In node {self.name}")
                assert False, "encountered a zero sum! contradiction in factor graph?"

            msgout[targetIdx, :] = p / s

            curridx[targetIdx] = True

        # damping at output
        if self.prevOut is not None:
            msgout[targets] = d*msgout[targets] + (1-d)*self.prevOut[targets]

        self.spreadOutgoing(msgout, targets)

        self.inlinenorm(msgin)

        self.prevOut = deepcopy(msgout)
        self.prevIn = deepcopy(msgin)

    def norm(self, msgin=None):
        Mm = self.gatherIncoming()
        Zn = np.prod(Mm,axis=0)
        s = np.sum(Zn)
        if s == 0:
            print(f"In norm of node {self.name}")
            assert s != 0, "encountered a zero sum! contradiction in factor graph?"
        P = Zn / s

        self.finalDist = P
        Px = self.finalDist[self.finalDist > 0] # have to avoid zeros, due to the logarithm
        self.entropy = np.maximum(-np.sum(np.multiply(Px, np.log2(Px))), 0)

        # "change" = statistical distance
        statistical_distance = np.max(np.abs(self.finalDist - self.prevDist))
        self.change = statistical_distance
        self.prevDist = np.copy(self.finalDist)

        self.value = np.argmax(self.finalDist)

        self.marginalize()

    def inlinenorm(self, msgin):
        Zn = np.prod(msgin,axis=0)
        s = np.sum(Zn)
        if s == 0:
            Zn = np.ones(Zn.shape)
        P = Zn / np.sum(Zn)
        self.finalDist = P
        Px = self.finalDist[self.finalDist > 0] # have to avoid zeros, due to the logarithm
        self.entropy = np.maximum(-np.sum(np.multiply(Px, np.log2(Px))), 0)

        self.value = np.argmax(self.finalDist)

    #marginalizes all bits, starts with final dist
    def marginalize(self):
        self.bitsDist = bitMarginalize(self.finalDist)
        self.bits = np.argmax(self.bitsDist, axis=1)

        for i in range(self.numbits):
            d = self.bitsDist[i, :]
            d = d[d > 0]
            self.bitsEntropy[i] = np.maximum(-np.sum(np.multiply(d, np.log2(d))), 0)



