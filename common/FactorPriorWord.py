from Node import Node
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

class FactorPriorWord(FactorGeneric, TraitPrior):
    def __init__(self,name,wordsize,dist):
        tlen = 2**wordsize
        assert(len(dist)==tlen)
        
        self.table = np.zeros((tlen, wordsize+1), dtype=settings.NUMPY_DATATYPE)

        # for 2^n columns --> all combinations from 0:2^n-1
        for i in range(tlen):
            self.table[i, :-1] = np.array(utils.intToBits(i, wordsize), dtype=settings.NUMPY_DATATYPE)
        
        # last column --> probability
        self.table[:, -1] = dist

        super().__init__(name)
