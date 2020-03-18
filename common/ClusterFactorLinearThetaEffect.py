
from Node import Node
from ClusterFactorLinear import ClusterFactorLinear
import numpy as np
import settings

# linear factor --> everything that can be written using just XORs (such as theta, parity, etc)
# this variant --> all messages are aligned! --> can be used for computing parity and for theta, but not for theta effect
class ClusterFactorLinearThetaEffect(ClusterFactorLinear):
    def __init__(self, name):
        super().__init__(name)

    # packed in own function to keep core the same
    def gatherIncoming(self):
        #edge 0 --> aligned input
        #edge 1 --> non-aligned, 7 bits, upper part
        #edge 2 --> non-aligned, 1 bit, lower part
        #edge 3 --> aligned output

        msgin = np.zeros(shape=(4, self.numvalues), dtype=settings.NUMPY_DATATYPE)
        msgin[0, :] = self.edges[0].m2f
        msgin[3, :] = self.edges[3].m2f

        msgin[1, ::2] = self.edges[1].m2f #only where LSB=0
        msgin[2, :2] = self.edges[2].m2f #set where all upper bits are 0, only lower bit either 0 or 1 (because xor with 0 has no effect)

        return msgin

    # packed in own function to keep core the same
    def spreadOutgoing(self, msgout):

        self.edges[0].m2n = msgout[0, :]
        self.edges[3].m2n = msgout[3, :]

        self.edges[1].m2n = msgout[1, ::2] #extract where LSB=0 (MSB=1 is not a valid solution here)
        self.edges[2].m2n = msgout[2, :2] #extract only lowest 2 cases, as upper bits must not be set
