from Node import Node
from FactorGeneric import FactorGeneric
import numpy as np
import settings

class FactorSbox(FactorGeneric):
    table0 = []
    table1 = []
    initialized = False

    @staticmethod
    def computeTables():
        chi = [0] * 32
        for i in range(32):
            N = [0,0,0,0,0]
            NN = [0,0,0,0,0]
            for j in range(5):
                N[j] = (i & (0x01 << j)) >> j
            for j in range(5):
                Fnand = (N[(j+1)%5] ^ 0x01) & N[(j+2)%5]
                NN[j] = Fnand ^ N[(j+0)%5]
            ii = 0
            for j in range(5):
                ii = ii | (NN[j] << j)
            chi[i] = ii


        for input in range(2**5):
            for output in range(2**5):
                in_bits = [input >> i & 1 for i in range(5)]
                out_bits = [output >> i & 1 for i in range(5)]
                if output == chi[input]:
                    prob = [1]

                    inout_bits = in_bits + out_bits + prob
                    FactorSbox.table0.append(inout_bits)
                    # flip output bit if iota is 1
                    out_bits[0] = out_bits[0]^1

                    inout_bits = in_bits + out_bits + prob
                    FactorSbox.table1.append(inout_bits)

        FactorSbox.table0 = np.array(FactorSbox.table0, dtype=settings.NUMPY_DATATYPE)
        FactorSbox.table1 = np.array(FactorSbox.table1, dtype=settings.NUMPY_DATATYPE)

        FactorSbox.initialized = True

    def __init__(self,name, iota):
        # [in0, in1, in2, in3, in4, out0, out1, out2, out3, out4, Prob]

        if not FactorSbox.initialized:
            FactorSbox.computeTables()
        
        if iota:
            self.table = FactorSbox.table1
        else:
            self.table = FactorSbox.table0

        super().__init__(name)

    def __repr__(self):
        return f"FactorSbox {self.name}"

