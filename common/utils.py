import matplotlib.pyplot as plt; plt.rcdefaults()
def printChi(G):
    G.norm()
    plt.subplot(2,5,1)
    for t in range(1,4):
        for z in range(1):
            for y in range(1):
                for x in range(5):
                    plt.subplot(3,5,x+5*(t-1)+1)
                    plt.bar([0,1],G.getNode(f"N{x}{y}{z}{t}").finalDist, align='center')
                    plt.xticks([0,1], [0,1])
                    plt.xlabel('')
                    plt.yticks([])
                    plt.ylabel('')
    plt.savefig(f"fig/{G.iter}")
    plt.close()

def bitToDist(bit):
    return [0.0,1.0] if bit == 1 else [1.0,0.0]

def intToBits(val, wordsize):
    return [val >> i & 1 for i in range(wordsize)]

def bitsToInt(bitlist):
    bits = np.array(bitlist)
    return np.polyval(bits[::-1], 2)

import numbers
def isNumber(x):
    return isinstance(x, numbers.Number)

# generic naming template
def gName(pref, x, y, z, t):
    if isNumber(x): x = f"{x:1d}"
    if isNumber(y): y = f"{y:1d}"
    if isNumber(z): z = f"{z:02d}"
    if isNumber(t): t = f"{t:02d}"

    return f"{pref}_{x}_{y}_{z}_{t}"

# naming template for nodes
def nName(x, y, z, t):
    return gName("N", x, y, z, t)

# naming template for deterministic factors
def fName(x, y, z, t):
    return gName("F", x, y, z, t)

# naming template for prior factors
def pName(x, y, z, t):
    return gName("P", x, y, z, t)

# naming template for leak factors (leak factors are a subset of prior factors)
def lName(x, y, z, t):
    return gName("L", x, y, z, t)

# generic marginalization function
import numpy as np
def bitMarginalize(probin):
    bitlen = np.log2(len(probin)).astype(int)
    probout = np.zeros(shape=(bitlen, 2))
    for i in range(bitlen):
        probin = np.reshape(probin, newshape=(-1, 2))
        probout[i, :] = np.sum(probin, axis=0)
        probin = np.sum(probin, axis=1)
    return probout

def popcount(x):
    x -= (x >> 1) & 0x5555555555555555
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
    return ((x * 0x0101010101010101) & 0xffffffffffffffff ) >> 56
