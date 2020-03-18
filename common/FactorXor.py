from Node import Node
from FactorLinear import FactorLinear
import numpy as np
import settings
##################
#                #
#  (A)           #
#     \          #
#     [XOR]---(C)  #
#    /           #
#  (B)           #
#                #
##################

# An XOR is the simplest form of the linear factor (2 inputs, output)
class FactorXor(FactorLinear):
    def __init__(self,name):
        super().__init__(name)
