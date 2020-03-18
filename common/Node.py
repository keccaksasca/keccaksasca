import numpy as np
from utils import nName
from copy import deepcopy

from TraitPrior import TraitPrior
import settings
from BPEnums import *

class Node(object):
    def __init__(self, x, y, z, t):
        self.edges = []
        self.finalDist = []
        self.entropy = 1
        self.prevDist = np.array([0.5, 0.5])
        self.change = 0
        # coordinates stored separately, for easier filtering etc.
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.name = nName(x, y, z, t)

        self.prevMsg = None
        self.prevP0 = None
        self.prevP1 = None

        self.bit = 0

        self.IODirections = []

        self.inFactors = []
        self.outFactors = []
        self.priorFactors = []

    def __repr__(self):
        return f"Node {self.name}: ({self.finalDist[0]:.2f}, {self.finalDist[1]:.2f})"

    def initMessages(self):
        for edge in self.edges:
            edge.m2f = np.array([0.5, 0.5])

        self.IODirections = np.array(self.IODirections)
        self.inFactors = np.flatnonzero(self.IODirections == IODirection.In)
        self.outFactors = np.flatnonzero(self.IODirections == IODirection.Out)
        self.priorFactors = np.flatnonzero(self.IODirections == IODirection.Prior)

        l = len(self.edges)

        self.prevMsg = np.full(shape=(l, 2), fill_value=0.5, dtype=settings.NUMPY_DATATYPE)
        self.prevP0 = np.full(shape=l, fill_value=0.5, dtype=settings.NUMPY_DATATYPE)
        self.prevP1 = np.full(shape=l, fill_value=0.5, dtype=settings.NUMPY_DATATYPE)

    def gatherIncoming(self):
        msgin = np.zeros(shape=self.prevMsg.shape, dtype=settings.NUMPY_DATATYPE)
        for (idx, edge) in enumerate(self.edges):
            msgin[idx, :] = edge.m2n

        return msgin


    def n2f(self, direction=None):
        l = len(self.edges)

        msgin = self.gatherIncoming()


        # there is a previous message --> use damping
        d = settings.DAMP
        msg = msgin

        tmp0 = np.tile(msg[:, 0], (l, 1))
        np.fill_diagonal(tmp0, 1.0)
        p0 = np.prod(tmp0, axis=1)

        tmp1 = np.tile(msg[:, 1], (l, 1))
        np.fill_diagonal(tmp1, 1.0)
        p1 = np.prod(tmp1, axis=1)

        s = np.add(p0, p1)
        if np.any(s==0):
            print(f"{self.name}: encountered a zero sum! contradiction in factor graph?")
            assert False
        p0 = np.divide(p0, s)
        p1 = np.divide(p1, s)

        #damping at output
        p0 = d*p0 + (1-d)*self.prevP0
        p1 = d*p1 + (1-d)*self.prevP1

        for (idx, edge) in enumerate(self.edges):
            edge.m2f = [p0[idx], p1[idx]]

        self.inlinenorm(msg)

        if self.name in settings.watchnodes:
            t = 0 # debug line
        self.prevP0 = p0
        self.prevP1 = p1
        self.prevMsg = deepcopy(msgin)

    def norm(self):

        Mm = self.gatherIncoming()
        Zn = np.prod(Mm,axis=0)
        P = Zn / np.sum(Zn)

        self.finalDist = P
        Px = self.finalDist[self.finalDist != 0] # have to avoid zeros, due to the logarithm
        self.entropy = np.max(-np.sum(np.multiply(Px, np.log2(Px))), 0)

        # "change" = statistical distance
        # 0 and 1 change by the same amplitude, just in different directions --> suffices to just do one of them
        statistical_distance = np.abs(self.finalDist[0] - self.prevDist[0])
        self.change = statistical_distance
        self.prevDist = np.copy(self.finalDist)

        self.bit = np.argmax(self.finalDist)

    def inlinenorm(self, msgin):
        Zn = np.prod(msgin,axis=0)
        P = Zn / np.sum(Zn)
        self.finalDist = P
        Px = self.finalDist[self.finalDist != 0] # have to avoid zeros, due to the logarithm
        self.entropy = np.max(-np.sum(np.multiply(Px, np.log2(Px))), 0)

        self.bit = np.argmax(self.finalDist)
