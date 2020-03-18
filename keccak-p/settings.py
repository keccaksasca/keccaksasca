import numpy as np
from scipy.linalg import hadamard
from utils import intToBits, popcount


def init():
    global NUMPY_DATATYPE, DAMP, CLUSTERSIZE, watchnodes, watchfactors
    NUMPY_DATATYPE=np.float64
    # NUMPY_DATATYPE=np.float32
    DAMP=1.0
    CLUSTERSIZE=8

    watchnodes = []
    watchfactors = []

def precomp():
    global WMAT, HWTABLE, BITTABLE
    if CLUSTERSIZE <= 8:
        WMAT = hadamard(2**CLUSTERSIZE)
    else:
        WMAT = None
    
    HWTABLE = np.zeros(shape=2**CLUSTERSIZE, dtype=int)
    for i in range(2**CLUSTERSIZE):
        HWTABLE[i] = popcount(i)
    
    BITTABLE = np.zeros(shape=(2**CLUSTERSIZE, CLUSTERSIZE), dtype=np.bool)
    for i in range(2**CLUSTERSIZE):
        BITTABLE[i, :] = np.array(intToBits(i, CLUSTERSIZE))
