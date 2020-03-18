from Node import Node
from Factor import Factor
from Edge import Edge
from copy import deepcopy
import numpy as np
import settings

##################
#                #
#  (A)           #
#     \          #
#     [?]---(C)  #
#    /           #
#  (B)           #
#                #
##################



#####################################
# Table-based factor

class FactorGeneric(Factor):
    table = None #done to get rid of warning
    def __init__(self,name):
        #   [A,B,C,Prob]

        # table-based factors always have a table...
        self.ltable = (self.table[:, 0:-1] != 0)

        # print("[F] init: " + self.name)

        super().__init__(name)

    def f2n(self):
        l = len(self.edges)

        msgin = self.gatherIncoming()

        ###### new version using vectorized instructions
        #collect all messages
        # init array (could do any value besides 0.5 as well, just do it such that sum = 1)
        tfill = deepcopy(self.table)
        #now fill the table with the values from the messages
        # for nodeIdx,node in enumerate(nodes):
        for nodeIdx in range(l):
            m = msgin[nodeIdx, :]

            idx = self.ltable[:, nodeIdx]
            tfill[np.logical_not(idx), nodeIdx] = m[0]
            tfill[idx, nodeIdx] = m[1]


        idxall = np.full(l+1, True)
        #now perform multiplications for each target node
        # for targetIdx,target in enumerate(nodes):
        for (targetIdx, edge) in enumerate(self.edges):
            curridx = idxall.copy()
            curridx[targetIdx] = False

            #product of each row (entry in the lut)
            p = np.prod(tfill[:, curridx], 1)

            i1 = self.ltable[:, targetIdx]
            i0 = np.logical_not(i1)
            s0 = np.sum(p[i0])
            s1 = np.sum(p[i1])

            edge.m2n = np.array([s0, s1])

