from TraitPrior import TraitPrior
from FactorPriorBit import FactorPriorBit
from ClusterFactorPriorHard import ClusterFactorPriorHard
from Edge import Edge

import numpy as np
import time
from utils import isNumber
import typing
from BPEnums import *
import settings

class Graph(object):
    def __init__(self):
        self.factors = []
        self.factorDict = {}
        self.nodes = []
        self.nodeDict = {}
        self.nodesFiltT = []
        self.iter = 0
        self.debug = 0
        self.debugTime = 1

        self.scheduleVariable = []
        self.scheduleFactor = []
        self.schedulePrior = []

        if self.debug: print("[G] init")

    def addFactor(self,factor):
        if factor.name in self.factorDict:
          raise Exception(f"factor {factor.name} already in graph")
        self.factors.append(factor)
        self.factorDict[factor.name] = factor

    def addNode(self,node):
        if node.name in self.nodeDict:
          raise Exception(f"node {node.name} already in graph")
        self.nodes.append(node)
        self.nodeDict[node.name] = node

        t = node.t
        y = node.y
        if isNumber(y):
            l = len(self.nodesFiltT)
            if(t >= l):
                self.nodesFiltT += [[]*(t+1-l)]
            self.nodesFiltT[t] += [node]

    def getFactor(self,fname):
        return self.factorDict[fname]

    def getNode(self,nname):
        return self.nodeDict[nname]

    def conFactorNode(self,fname,nname, direction, nodeIndizes=None):
        targetFactor = self.getFactor(fname)
        targetNode = self.getNode(nname)
        assert(targetFactor is not None),"Specified factor \"" + fname + "\" does not exist!"
        assert(targetNode is not None),"Specified node \"" + nname + "\" does not exist!"

        e = Edge(targetNode, targetFactor, direction, nodeIndizes=nodeIndizes)

        targetFactor.edges.append(e)
        targetNode.edges.append(e)

        if self.debug: print("[G] connecting " + targetFactor.name + " <-> " + targetNode.name)

    def printFactorCon(self):
        print("[G] factor connections:")
        for f in self.factors:
            print("[G]   " + f.name + " ->")
            for n in f.conNodes:
                print("[G]     " + n.name)

    def printNodeCon(self):
        print("[G] node connections:")
        for n in self.nodes:
            print("[G]   " + n.name + " ->")
            for f in n.conFactors:
                print("[G]     " + f.name)

    def printRes(self):
        print("[G] results:")
        for n in self.nodes:
            print("[G]   " + n.name + " : " + str(n.finalDist))

    def norm(self):
        if self.debug: print("[G] normalizing")
        for n in self.nodes:
            n.norm()

    def __f2n(self):
        if self.debug: print("[G] msgs factor -> nodes:")
        times = {}
        calls = {}
        for f in self.factors:
            start = time.time()
            f.f2n()
            end = time.time()
            t = end - start

            n = f.__class__.__name__

            if n in times:
                times[n] += t
                calls[n] += 1
            else:
                times[n] = t
                calls[n] = 1

        if self.debugTime:
            for c in times:
                ttime = times[c]
                tcalls = calls[c]

                print('Factor: %s, called %d, total: %f s, avg: %f s'%(c, tcalls, ttime, ttime/tcalls))


    def __n2f(self):
        if self.debug: print("[G] msgs nodes -> factors:")

        start = time.time()

        for n in self.nodes:
            n.n2f()


        if self.debugTime:
            ntime = time.time() - start
            ncalls = len(self.nodes)
            print('Nodes, called %d, total: %.2f s, avg: %.2f us'%(ncalls, ntime, ntime/ncalls*1e6))

    def f2n2f(self):
        self.__f2n()
        self.__n2f()
        self.iter = self.iter + 1

    def n2f2n(self):
        self.__n2f()
        self.__f2n()
        self.iter = self.iter + 1

    def scheduledUpdate(self):

        self.iter += 1

        numlayers = len(self.scheduleFactor)

        #times for factors
        ftimes = {}
        fcalls = {}
        #time for variables
        ntime = 0

        #first layer variables treated separately
        for n in self.scheduleVariable[0]:
            n.n2f(target=IODirection.In)

        # forward: factor --> variable (--> prior --> variable)
        # factor: current layer, variable: next layer
        for r in range(numlayers):
            t = 0 # debug line
            hasPriors = (len(self.schedulePrior[r+1]) > 0)
            for f in self.scheduleFactor[r]:
                start = time.time()
                f.f2n()
                end = time.time()
                t = end - start

                n = f.__class__.__name__

                if n in ftimes:
                    ftimes[n] += t
                    fcalls[n] += 1
                else:
                    ftimes[n] = t
                    fcalls[n] = 1

            start = time.time()
            for n in self.scheduleVariable[r+1]:
                if hasPriors:
                    n.n2f(target = IODirection.Prior)
                else:
                    n.n2f(target = IODirection.In) #propagate to the INPUT of the functions (forwards)

            end = time.time()
            ntime += (end-start)

            if hasPriors: # dont't do anything if there are no priors for the current layer
                for f in self.schedulePrior[r+1]:
                    start = time.time()
                    f.f2n()
                    end = time.time()
                    t = end - start

                    n = f.__class__.__name__

                    if n in ftimes:
                        ftimes[n] += t
                        fcalls[n] += 1
                    else:
                        ftimes[n] = t
                        fcalls[n] = 1

                start = time.time()
                for n in self.scheduleVariable[r+1]:
                    n.n2f(target = IODirection.In)
                end = time.time()
                ntime += (end-start)

        if len(settings.watchnodes) > 0:
            print("------------------------------------------------")

        for n in self.scheduleVariable[-1]:
            n.n2f(target=IODirection.Out)

        # backward: factor --> variable (--> prior --> variable)
        # -1 : in last layer, there is only a variable and priors, no factors. Variable and priors were already updated in forward direction
        for r in reversed(range(numlayers)):
            hasPriors = (len(self.schedulePrior[r]) > 0)
            for f in self.scheduleFactor[r]:
                f.f2n()

            for n in self.scheduleVariable[r]:
                if hasPriors:
                    n.n2f(target=IODirection.Prior)
                else:
                    n.n2f(target=IODirection.Out) #propagate to the OUTPUT of the functions (backwards)

            if hasPriors: # dont't do anything if there are no priors for the current layer
                for f in self.schedulePrior[r]:
                    f.f2n()

                for n in self.scheduleVariable[r]:
                    n.n2f(target=IODirection.Out)

        if self.debugTime:
            ncalls = 2*len(self.nodes) # each variable is called twice (due to the priors etc)
            print('Nodes, called %d, total: %.2f s, avg: %.2f us'%(ncalls, ntime, ntime/ncalls*1e6))

            for c in ftimes:
                ttime = ftimes[c]
                tcalls = fcalls[c]

                print('Factor: %s, called %d, total: %.2f s, avg: %.2f us'%(c, tcalls, ttime, ttime/tcalls*1e6))


    def initMessages(self):
        for factor in self.factors:
            factor.initMessages()
        for node in self.nodes:
            node.initMessages()

    def spreadPriors(self):
        #hack stuff in here: filter out all factors which are leafs from schedule
        # a lot easier to do here than in addLeakage
        for i in range(len(self.schedulePrior)):
            self.schedulePrior[i] = [x for x in self.schedulePrior[i] if not x.isLeaf]


        for factor in self.factors:
            if isinstance(factor, FactorPriorBit) or isinstance(factor, ClusterFactorPriorHard):
                factor.f2n()

        for factor in self.schedulePrior[0]:
            factor.f2n()

        for node in self.filterNodesT(0):
            node.n2f()
            node.norm()

    def nodesProperty(self, func=sum, property="entropy", x=None, y=None, z=None, t=None):
        l = self.filterNodes(x, y, z, t)
        e = func(getattr(n, property) for n in l)
        return e

    def sumEntropyT(self, t, avg=False):
        l = self.filterNodesT(t)
        e = 0
        for n in l:
            e += n.entropy
        if avg:
            e = e/len(l)
        return e

    def filterNodes(self, x=None, y=None, z=None, t=None):
        l = self.nodes
        if x is not None:
            l = [a for a in l if a.x == x]
        if y is not None:
            l = [a for a in l if a.y == y]
        else: # default --> only full states
            l = [a for a in l if isNumber(a.y)]
        if z is not None:
            l = [a for a in l if a.z == z]
        if t is not None:
            l = [a for a in l if a.t == t]
        return l

    def filterNodesT(self, t):
        return self.nodesFiltT[t]
