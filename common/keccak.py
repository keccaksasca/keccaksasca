import sys

import random
import numpy as np
import settings
import utils

from scipy.stats import norm

LANE_LENGTH = 64

pix = []
piy = []
rhoz = []
iota = []
irhoz = []
ipix = []
ipiy = []

def setLaneLength(lane_length):
    global LANE_LENGTH
    LANE_LENGTH = lane_length
    computeConstants()

def randomState():
    return [[[random.randint(0,1) for z in range(LANE_LENGTH)] for y in range(5)] for x in range(5)]

def emptyState():
    return [[[0 for z in range(LANE_LENGTH)] for y in range(5)] for x in range(5)]

def oneState():
    return [[[1 for z in range(LANE_LENGTH)] for y in range(5)] for x in range(5)]

def printState(state):
  for sliceidx in range(LANE_LENGTH):
      print(f"###slice {sliceidx}###")
      for rowidx in range(5):
        row = [state[x][rowidx][sliceidx] for x in range(5)]
        print(f"{row[0]}{row[1]}{row[2]}{row[3]}{row[4]}")

# returns probabilities for hamming weights
def wordLeak(word, sigma):
    # compute the hamming weight of the word
    wordhw = sum(word)
    # generate leakage (hw + gaussian)
    wordleak = wordhw + np.random.normal(0, sigma, 1)

    wordsize = len(word)
    # match
    hws = range(0, wordsize+1)
    p = norm.pdf(wordleak, hws, sigma)
    return p

#Author: A.Polino (https://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/)
def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)

# takes as input the state and returns the leakage
def stateLeakHW(state, wordsize, sigma):
    assert is_power2(wordsize) and wordsize <= LANE_LENGTH, "Wordsize must be a power of two and smaller than the lane length"

    # array that stores probabilities of hamming weights for the entire state (one entry per word)
    leak = np.zeros((5, 5, LANE_LENGTH//wordsize, wordsize+1), dtype=settings.NUMPY_DATATYPE)

    for x in range(5):
        for y in range(5):
            for z in range(0, LANE_LENGTH, wordsize):
                # bunch together the word
                word = [state[x][y][z+zz] for zz in range(wordsize)]

                leak[x, y, z//wordsize, :] = wordLeak(word, sigma)
    return leak

def stateClustered(state, wordsize):
    assert is_power2(wordsize) and wordsize <= LANE_LENGTH, "Wordsize must be a power of two and smaller than the lane length"

    leak = np.zeros((5, 5, LANE_LENGTH//wordsize), dtype=np.uint64)

    for x in range(5):
        for y in range(5):
            for z in range(0, LANE_LENGTH, wordsize):
                # bunch together the word
                bits = [state[x][y][z+zz] for zz in range(wordsize)]
                leak[x, y, z//wordsize] = utils.bitsToInt(bits)

    return leak


# takes as input the state and returns the leakage
def planeLeakHW(plane, wordsize, sigma):
    assert is_power2(wordsize) and wordsize <= LANE_LENGTH, "Wordsize must be a power of two and smaller than the lane length"

    # array that stores probabilities of hamming weights for the entire state (one entry per word)
    leak = np.zeros((5, LANE_LENGTH//wordsize, wordsize+1), dtype=settings.NUMPY_DATATYPE)

    for x in range(5):
        for z in range(0, LANE_LENGTH, wordsize):
            # bunch together the word
            word = [plane[x][z+zz] for zz in range(wordsize)]

            leak[x, z//wordsize, :] = wordLeak(word, sigma)

    return leak

def planeClustered(plane, wordsize):
    assert is_power2(wordsize) and wordsize <= LANE_LENGTH, "Wordsize must be a power of two and smaller than the lane length"

    # array that stores probabilities of hamming weights for the entire state (one entry per word)
    leak = np.zeros((5, LANE_LENGTH//wordsize), dtype=np.uint64)

    for x in range(5):
        for z in range(0, LANE_LENGTH, wordsize):
            # bunch together the word
            bits = [plane[x][z+zz] for zz in range(wordsize)]

            leak[x, z//wordsize] = utils.bitsToInt(bits)

    return leak

##########################################
#                  THETA                 #
##########################################

def ParityPlane(state):
    def parity(column):
        p = 0
        for v in column:
            p ^= v
        return p

    # compute the parity plane
    parityPlane =  [[0 for z in range(LANE_LENGTH)] for x in range(5)]
    for x in range(5):
        for z in range(LANE_LENGTH):
            column = [state[x][y][z] for y in range(5)]
            parityPlane[x][z] = parity(column)

    return parityPlane

# compute Theta to check output of Graph
def Theta(state):
    parityPlane = ParityPlane(state)

    # for each coordinate in the parity plane, compute the theta effect and XOR it to every bit in the column
    for z in range(LANE_LENGTH):
        for x in range(5):
            p0 = parityPlane[(x+1)%5][(z-1)%LANE_LENGTH]
            p1 = parityPlane[(x-1)%5][z]
            theta_effect = p0^p1

            for y in range(5):
                state[x][y][z] ^= theta_effect

    return state

##########################################
#                 RHOPI                  #
##########################################

def RhoPi(state):
    state_out = emptyState()
    for z in range(LANE_LENGTH):
        for y in range(5):
            for x in range(5):
                state_out[pix[x][y]][piy[x][y]][(rhoz[x][y]+z)%LANE_LENGTH] = state[x][y][z]
    return state_out

##########################################
#                  CHI                   #
##########################################

def Chi(state):
    for z in range(LANE_LENGTH):
        for y in range(5):
            N = [state[0][y][z],state[1][y][z],state[2][y][z],state[3][y][z],state[4][y][z]]
            NN = [0,0,0,0,0]
            for j in range(5):
                Fnand = (N[(j+1)%5]^0x01)&N[(j+2)%5]
                NN[j] = Fnand^N[(j+0)%5]
            state[0][y][z] = NN[0]
            state[1][y][z] = NN[1]
            state[2][y][z] = NN[2]
            state[3][y][z] = NN[3]
            state[4][y][z] = NN[4]
    return state

##########################################
#                 IOTA                   #
##########################################


def Iota(state,round):
    for z in range(LANE_LENGTH):
        state[0][0][z] = state[0][0][z] ^ iota[round][z]
    return state



def computeConstants():
    global pix, piy, rhoz, iota, irhoz, ipix, ipiy
    pix = np.zeros((5,5),dtype=int)
    piy = np.zeros((5,5),dtype=int)
    for y in range(5):
        for x in range(5):
            pix[x,y] =  y
            piy[x,y] = (2*x + 3*y) % 5
    rhoz = [
        [0%LANE_LENGTH,1%LANE_LENGTH,190%LANE_LENGTH,28%LANE_LENGTH,91%LANE_LENGTH],
        [36%LANE_LENGTH,300%LANE_LENGTH,6%LANE_LENGTH,55%LANE_LENGTH,276%LANE_LENGTH],
        [3%LANE_LENGTH,10%LANE_LENGTH,171%LANE_LENGTH,153%LANE_LENGTH,231%LANE_LENGTH],
        [105%LANE_LENGTH,45%LANE_LENGTH,15%LANE_LENGTH,21%LANE_LENGTH,136%LANE_LENGTH],
        [210%LANE_LENGTH,66%LANE_LENGTH,253%LANE_LENGTH,120%LANE_LENGTH,78%LANE_LENGTH]
    ]
    rhoz=[*zip(*rhoz)]

    irhoz = np.array(rhoz, dtype=int)
    irhoz = (-irhoz) % LANE_LENGTH
    irhoz = irhoz.tolist()

    pixy = pix + 5*piy
    ipixy = np.zeros(25, dtype=int)
    ipixy[pixy.flatten(order='F')] = np.arange(25) # for flatten, need to use column-major order (x + )
    ipixy = np.reshape(ipixy, newshape=(5, 5), order='F')

    ipix = ipixy % 5
    ipiy = ipixy // 5

    iota_64 = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008
    ]
    iota = np.zeros((24,LANE_LENGTH),dtype=int)
    for r in range(24):
        bitstr = "{0:b}".format(iota_64[r] & int("1"*LANE_LENGTH,2)).zfill(LANE_LENGTH)[::-1]
        for idx,c in enumerate(bitstr):
            iota[r][idx] = 1 if (c == '1') else 0

computeConstants()


##########################################
#                 ROUND                  #
##########################################

def round(state, roundIdx):
  state = Theta(state)
  state = RhoPi(state)
  state = Chi(state)
  state = Iota(state,roundIdx)
  return state

def keccak_permutation(state):
  for roundIdx in range(24):
    state = round(state, roundIdx)
  return state



if __name__== "__main__":
  state = randomState()
  print("random state:")
  print(state)
  state_t = Theta(state)
  print("AFTER THETA:")
  print(state_t)
  state_tpr = RhoPi(state_t)
  print("AFTER THETA-PI-RHO:")
  print(state_tpr)
  state_tprc = Chi(state_tpr)
  print("AFTER THETA-PI-RHO-CHI:")
  print(state_tprc)
  state_tprci = Iota(state_tprc,0)
  print("AFTER THETA-PI-RHO-CHI-IOTA:")
  print(state_tprci)
