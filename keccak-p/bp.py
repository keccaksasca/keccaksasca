import sys
sys.path.append('../common/')

from Graph import Graph
from ClusterNode import ClusterNode
from BPEnums import IODirection

# Factors used
from FactorPriorBit import FactorPriorBit
from FactorPriorWordHW import FactorPriorWordHW
from FactorXor import FactorXor
from FactorSbox import FactorSbox

from ClusterFactor import ClusterFactor
from ClusterFactorLinear import ClusterFactorLinear
from ClusterFactorLinearThetaEffect import ClusterFactorLinearThetaEffect
from ClusterFactorPriorHW import ClusterFactorPriorHW
from ClusterFactorPriorHard import ClusterFactorPriorHard

import utils
from utils import nName, pName, fName, lName, bitToDist
import keccak
from plotting import *
import settings
import time
import random
from copy import deepcopy

import numpy as np
import argparse

settings.init()

r'''
 t = time
 y   z
 |  /
 | /
 * ---- x
                  . . . . . . . . . . . . . . . . . . . . . . . .
               .                                             .  .
            .   (N0410) (N1410) (N2410) (N3410) (N4410)   .     .
         .                                             .        .
      .   (N0400) (N1400) (N2400) (N3400) (N4400)   .   (N4410) .
   .                                             .              .
. . . . . . . . . . . . . . . . . . . . . . . .         (N4310) .
.                                             . (N4400)         .
.   (N0400) (N1400) (N2400) (N3400) (N4400)   .         (N4210) .
.                                             . (N4300)         .
.   (N0300) (N1300) (N2300) (N3300) (N4300)   .         (N4110) .
.                                             . (N4200)         .
.   (N0200) (N1200) (N2200) (N3200) (N4200)   .         (N4010) .
.                                             . (N4100)      .
.   (N0100) (N1100) (N2100) (N3100) (N4100)   .           .
.                                             . (N4000).
.   (N0000) (N1000) (N2000) (N3000) (N4000)   .     .
. . . . .                                     .  .
N(xyzt) . . . . . . . . . . . . . . . . . . . .

'''


# global vars
pix = []
piy = []
rhoz = []
iota = []
irhoz = []
ipix = []
ipiy = []


def computeConstants(lane_length):
    global pix, piy, rhoz, iota, ipix, ipiy, irhoz
    keccak.setLaneLength(lane_length)
    pix = keccak.pix
    piy = keccak.piy
    rhoz = keccak.rhoz
    iota = keccak.iota
    ipix = keccak.ipix
    ipiy = keccak.ipiy
    irhoz = keccak.irhoz


def clusteridx(idx):
    c1 = idx%settings.CLUSTERSIZE
    c2 = idx - c1
    return(c2, [c1])


def build_graph(lane_length, num_rounds):
    print("Starting construction of factor graph")

    ############################
    # ADDING NODES AND FACTORS #
    ############################

    graph = Graph()

    clustersize = settings.CLUSTERSIZE

    #TODO system to automate message schedule

    print("Adding nodes")

    fullStateVariables = []
    # adding nodes for full state: before theta (after chi), after theta (before chi), output
    for t in range(1+2*num_rounds):
        currNodes = []

        for z in range(0, lane_length, clustersize):
            for y in range(5):
                for x in range(5):
                    n = ClusterNode(x, y, z, t, clustersize)
                    graph.addNode(n)
                    currNodes += [n]

        fullStateVariables += [currNodes]

    parityVariables = []
    thetaEffectVariables = []
    # adding nodes for: parity plane, theta-effect plane
    for t in range(num_rounds):
        currParityVariables = []
        currThetaEffectVariables = []
        for z in range(0, lane_length, clustersize):
            for x in range(5):
                # parity nodes
                n = ClusterNode(x, "p", z, t, clustersize)
                graph.addNode(n)
                currParityVariables += [n]
                # theta-effect nodes
                n = ClusterNode(x, "t", z, t, clustersize)
                graph.addNode(n)
                currThetaEffectVariables += [n]

        parityVariables += [currParityVariables]
        thetaEffectVariables += [currThetaEffectVariables]

    for t in range(num_rounds):
        graph.scheduleVariable += [fullStateVariables[2*t]]
        graph.scheduleVariable += [parityVariables[t]]
        graph.scheduleVariable += [thetaEffectVariables[t]]
        graph.scheduleVariable += [fullStateVariables[2*t+1]]

    graph.scheduleVariable += [fullStateVariables[-1]]

    ############################
    #       THETA+PI+RHO       #
    ############################

    r'''
     add parity of columns a[x-1][:][z] and a[x+1][:][z-1] to a[x][y][z]

     # a[x-1][:][z]                    ||   # a[x-1][:][z-1]
     [F00]-(N00)- \                    ||   [F10]-(N10)- \
                   [Fx01]-(Nx01)       ||                 [Fx11]-(Nx11)
     [F01]-(N01)- /       |            ||   [F11]-(N11)- /       |
                          |            ||                        |
     [F02]-(N02)------[Fx02]-(Nx02)    ||   [F12]-(N12)------[Fx12]-(Nx12)
                            /          ||                          /
     [F03]-(N03)------[Fx03]-(Nx03)    ||   [F13]-(N13)------[Fx13]-(Nx13)
                            /          ||                          /
     [F04]-(N04)------[Fx04]-(Nx04)    ||   [F14]-(N14)------[Fx14]-(Nx14)
                                |                                     |
                                |                                     |
     [F2]-(N2)-----------------[Fx20]----(N20)----------------------[Fx21]
                                                                      |
                                                                    (N21)
    '''

    t=0

    for r in range(num_rounds):
        print(f"Factors for round: {r}")

        currFactors = []

        # parity factors
        for z in range(0, lane_length, clustersize):
            zx = clusteridx(z)
            for x in range(5):
                fname = fName(x, "p", z, r)
                f = ClusterFactorLinear(fname) # was: FactorParity
                graph.addFactor(f)
                currFactors += [f]
                for y in range(5):
                    graph.conFactorNode(fname, nName(x, y, z, t), IODirection.In)
                graph.conFactorNode(fname, nName(x, "p", z, r), IODirection.Out)

        graph.scheduleFactor += [currFactors]
        currFactors = []

        # connect parity factors to theta-effect factors
        for z in range(0, lane_length, clustersize):
            # zx = clusteridx(z)
            for x in range(5):
                fname = fName(x, "t", z, r)

                if clustersize > 1:
                    f = ClusterFactorLinearThetaEffect(fname) #was: FactorXor
                    graph.addFactor(f)
                    currFactors += [f]

                    #1st edge --> alinged input
                    graph.conFactorNode(fname, nName((x-1)%5, "p", z, r), IODirection.In)
                    #2nd edge --> non-aligned, 7 bits, upper part
                    graph.conFactorNode(fname, nName((x+1)%5, "p", z, r), IODirection.In, np.arange(clustersize-1)) #use the lower parts of this node
                    #3rd edge --> non-aligned, 1 bit, lower part
                    graph.conFactorNode(fname, nName((x+1)%5, "p", (z - clustersize) % lane_length, r), IODirection.In, np.array([clustersize-1])) #use the MSB of this node
                    #4th edge --> aligned output
                    graph.conFactorNode(fname, nName(x, "t", z, r), IODirection.Out)
                else:
                    f = FactorXor(fname)
                    graph.addFactor(f)
                    currFactors += [f]
                    # first column parity
                    graph.conFactorNode(fname, nName((x+1)%5, "p", (z-1)%lane_length, r), IODirection.In)
                    # second column parity
                    graph.conFactorNode(fname, nName((x-1)%5, "p", z, r), IODirection.In)
                    # output bit
                    graph.conFactorNode(fname, nName(x, "t", z, r), IODirection.Out)


        graph.scheduleFactor += [currFactors]
        currFactors = []

        # connect theta-effect factors to theta
        for z in range(0, lane_length, clustersize):
            for y in range(5):
                for x in range(5):
                    fname = fName(x, y, z, t)
                    f = ClusterFactorLinear(fname) # was: FactorTheta
                    graph.addFactor(f)
                    currFactors += [f]
                    # theta effect
                    graph.conFactorNode(fname, nName(x, "t", z, r), IODirection.In)
                    # input bit
                    graph.conFactorNode(fname,nName(x, y, z, t), IODirection.In)
                    # output bit
                    graph.conFactorNode(fname,nName(x, y, z, t+1), IODirection.Out)

        graph.scheduleFactor += [currFactors]
        currFactors = []

        t += 1

        ############################
        #        CHI+IOTA          #
        ############################

        r'''
        (N0yzt) (N1yzt) (N2yzt) (N3yzt) (N4yzt)
         |       |       |       :      :
         |        \     /
         |         \   /
         |        [F0yzta]
         |           |
        [F0yztx]--(N0yz{t+1})
         |
         |               :           :           :           :
        (N0yz{t+2}) (N1yz{t+2}) (N2yz{t+2}) (N3yz{t+2}) (N4yz{t+2})
        '''

        for z in range(lane_length):
            zx = clusteridx(z)
            for y in range(5):
                fname = fName("c", y, z, t)
                f = FactorSbox(fname, 1 if (y==0) and (iota[r][z]==1) else 0)
                graph.addFactor(f)
                currFactors += [f]

                # connect inputs
                for x in range(5):
                    #implement Rho and Pi implicitly, through wiring --> since we do it here, we need to use the inverse permutations
                    ix = ipix[x, y]
                    iy = ipiy[x, y]
                    ir = irhoz[ix][iy]
                    zx1 = clusteridx((ir+z)%lane_length)
                    graph.conFactorNode(fname,nName(ix, iy, zx1[0], t), IODirection.In, zx1[1])

                # connect outputs
                for x in range(5):
                    graph.conFactorNode(fname, nName(x, y, zx[0], t+1), IODirection.Out, zx[1])


        graph.scheduleFactor += [currFactors]
        t += 1
    return graph, t


def add_leakage(graph, initial_state, initial_mask, final_mask, lane_length, num_rounds, word_size, sigma, noleak=False, chileak=False):
    print("Simulate leakage and add it to the Graph")

    t = 0

    state = deepcopy(initial_state)

    cur_factors = []

    clustersize = settings.CLUSTERSIZE

    assert word_size >= clustersize, "clusters must be smaller or equal to the wordsize"
    assert word_size % clustersize == 0, "wordsize must be an integer multiple of the clustersize"

    lhw = keccak.stateLeakHW(state, word_size, sigma)
    for x in range(5):
        for y in range(5):
            for z in range(0, lane_length, word_size):
                word_mask = initial_mask[x, y, z:(z + word_size)]

                if (not np.all(word_mask)) and (not noleak): #some bits of the current word are unknown --> add leakage
                    fname = lName(x, y, z, t)
                    dist = lhw[x, y, z // word_size, :]
                    f = ClusterFactorPriorHW(fname, word_size, dist) # was: FactorPriorWordHW
                    graph.addFactor(f)
                    cur_factors += [f]
                    for zz in range(0, word_size, clustersize):
                        graph.conFactorNode(fname, nName(x, y, z+zz, t), IODirection.Prior)

                # hard priors for known bits, no need to add them to schedule
                for zz in range(0, word_size, clustersize):
                    zx = z + zz
                    cluster_mask = initial_mask[x, y, zx:(zx + clustersize)]
                    if np.all(cluster_mask): # all bits of cluster are known --> ClusterFactorPriorHard
                        fname = pName(x, y, z+zz, t)
                        bits = state[x][y][(z+zz):(z+zz+clustersize)]
                        val = np.polyval(bits[::-1], 2)
                        graph.addFactor(ClusterFactorPriorHard(fname, clustersize, val))
                        graph.conFactorNode(fname, nName(x, y, z+zz, t), IODirection.Prior)
                    elif np.any(cluster_mask): # some bits are unknown --> bit priors
                        for zzz in range(clustersize):
                            zx = z + zz + zzz
                            if initial_mask[x, y, zx]:
                                fname = pName(x, y, zx, t)
                                dist = bitToDist(state[x][y][zx])
                                graph.addFactor(FactorPriorBit(fname, dist))
                                graph.conFactorNode(fname, nName(x, y, z+zz, t), IODirection.Prior, [zzz])

    graph.schedulePrior += [cur_factors]
    cur_factors = []

    for r in range(num_rounds):
        print(f"Priors for round: {r}")

        # parity leakage
        pPlane = keccak.ParityPlane(state)

        # word-wise (hamming weights)
        if not noleak:
            lhw = keccak.planeLeakHW(pPlane, word_size, sigma)
            for x in range(5):
                for z in range(0, lane_length, word_size):
                    fname = lName(x, "p", z, r)
                    dist = lhw[x, z // word_size, :]
                    f = ClusterFactorPriorHW(fname, word_size, dist) # was: FactorPriorWordHW
                    graph.addFactor(f)
                    cur_factors += [f]
                    for zz in range(0, word_size, clustersize):
                        graph.conFactorNode(fname, nName(x, "p", z+zz, r), IODirection.Prior)

        graph.schedulePrior += [cur_factors]

        cur_factors = []
        # no Leakage for the theta-effect plane --> add empty list
        graph.schedulePrior += [cur_factors]
        cur_factors = []

        t += 1

        # theta

        state = keccak.Theta(state)

        # leakage at output of theta
        if not noleak:
            lhw = keccak.stateLeakHW(state, word_size, sigma)
            for x in range(5):
                for y in range(5):
                    for z in range(0, lane_length, word_size):
                        fname = lName(x, y, z, t)+ "_0"
                        dist = lhw[x, y, z//word_size, :]
                        f = ClusterFactorPriorHW(fname, word_size, dist) # was: FactorPriorWordHW
                        graph.addFactor(f)
                        cur_factors += [f]
                        for zz in range(0, word_size, clustersize):
                            graph.conFactorNode(fname, nName(x, y, z+zz, t), IODirection.Prior)


        # rho + pi

        state = keccak.RhoPi(state)

        #generate leakage regardless of chileak, just to have same rng state
        lhw = keccak.stateLeakHW(state, word_size, sigma)

        # leakage at input of chi
        if chileak and not noleak:
            for x in range(5):
                for y in range(5):
                    ix = ipix[x, y]
                    iy = ipiy[x, y]
                    ir = irhoz[ix][iy]
                    for z in range(0, lane_length, word_size):
                        fname = lName(x, y, z, t)+ "_1"
                        dist = lhw[x, y, z // word_size, :]
                        f = ClusterFactorPriorHW(fname, word_size, dist) # was: FactorPriorWordHW
                        graph.addFactor(f)
                        cur_factors += [f]

                        iz = (ir + z) % lane_length
                        izz = np.arange(iz, iz + word_size) % lane_length
                        iz_node = izz % clustersize
                        iz_cluster = izz - iz_node
                        clusters = np.unique(iz_cluster)

                        for cluster in clusters:
                            nodeidx = iz_node[iz_cluster == cluster]
                            graph.conFactorNode(fname, nName(ix, iy, cluster, t), IODirection.Prior, nodeidx)

        graph.schedulePrior += [cur_factors]
        cur_factors = []

        t += 1

        # chi + iota

        state = keccak.Chi(state)
        state = keccak.Iota(state, r)

        # TODO leakage at the output of chi (without iota) --> iota flips some bits in lane (0, 0)

        # adding leakage after chi, if last round then perfect information is used based on final_mask
        lhw = keccak.stateLeakHW(state, word_size, sigma)
        for x in range(5):
            for y in range(5):
                for z in range(0, lane_length, word_size):
                    word_mask = final_mask[x, y, z:(z + word_size)]

                    # check if we have to add leakage
                    if ((r < (num_rounds - 1)) or (not np.all(word_mask))) and not noleak:
                        fname = lName(x, y, z, t)
                        dist = lhw[x, y, z // word_size, :]
                        f = ClusterFactorPriorHW(fname, word_size, dist) # was: FactorPriorWordHW
                        graph.addFactor(f)
                        cur_factors += [f]
                        for zz in range(0, word_size, clustersize):
                            graph.conFactorNode(fname, nName(x, y, z+zz, t), IODirection.Prior)


                    # check if we have to add bit priors at the output
                    if (r == num_rounds-1) and np.any(word_mask):
                        for zz in range(0, word_size, clustersize):
                            zx = z + zz
                            cluster_mask = final_mask[x, y, zx:(zx + clustersize)]
                            if np.all(cluster_mask): # all bits of cluster are known --> ClusterFactorPriorHard
                                fname = pName(x, y, z+zz, t)
                                bits = state[x][y][(z+zz):(z+zz+clustersize)]
                                val = np.polyval(bits[::-1], 2)
                                graph.addFactor(ClusterFactorPriorHard(fname, clustersize, val))
                                graph.conFactorNode(fname, nName(x, y, z+zz, t), IODirection.Prior)
                            elif np.any(cluster_mask): # some bits are unknown --> bit priors
                                for zzz in range(clustersize):
                                    zx = z + zz + zzz
                                    if final_mask[x, y, zx]:
                                        fname = pName(x, y, zx, t)
                                        dist = bitToDist(state[x][y][zx])
                                        graph.addFactor(FactorPriorBit(fname, dist))
                                        graph.conFactorNode(fname, nName(x, y, z+zz, t), IODirection.Prior, [zzz])

        graph.schedulePrior += [cur_factors]
        cur_factors = []

    return

def hex2keccak(hex_string, lane_length, dtype):
    allowed = set('0123456789abcdefABCDEFR')
    if not set(hex_string).issubset(allowed):
        hex_string = eval(hex_string)

    if len(hex_string) == 1:
        hex_string = hex_string*(5*5*lane_length//4)

    assert len(hex_string) == 2*25*lane_length//8, "incorrect mask length, must be exactly the state size"

    # replace 'R' in hex string with random values
    if "R" in hex_string:
        hex_string = "".join([x if x != 'R' else hex(np.random.randint(0, 16))[2].upper() for x in hex_string])
        print(f'state replaced: {hex_string}')

    return np.reshape(
        ([int(digit) for digit in format(int(hex_string, base=16), '0'+str(lane_length*25)+'b')]),
        (5, 5, lane_length)
    ).transpose((1, 0, 2)).astype(dtype)


# settings.watchnodes = []
# settings.watchfactors = []
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="custom")
    parser.add_argument("-l", "--lanelength", type=int, default=64)
    parser.add_argument("-r", "--rounds", type=int, default=2)  # default is set later
    parser.add_argument("--istate", type=str, default="999")  # default is set later
    parser.add_argument("--imask", type=str, default="999")  # default is set later
    parser.add_argument("--fmask", type=str, default="999")  # default is set later
    parser.add_argument("-w", "--wordsize", type=int, default=8)
    parser.add_argument("-s", "--sigma", type=float, default=1.0)
    parser.add_argument("-i", "--iterations", type=int, default=50)
    parser.add_argument("-b", "--blocking", action="store_true")
    parser.add_argument("-d", "--damping", type=float, default=0.75)  # 1: no damping
    parser.add_argument("-c", "--clustersize", type=int, default=8)
    parser.add_argument("-p", "--plots", action="store_true")
    parser.add_argument("--nochileak", action="store_true")
    parser.add_argument("--noleak", action="store_true")
    parser.add_argument("--seed", type=str, default=None)
    args = parser.parse_args()

    ####################################
    #        FETCH CMD PARAMS          #
    ####################################

    name = args.name
    lane_length = args.lanelength
    rounds = args.rounds
    istate = args.istate
    imask = args.imask
    fmask = args.fmask
    word_size = args.wordsize
    sigma = args.sigma
    iterations = args.iterations
    settings.DAMP = args.damping
    settings.CLUSTERSIZE = args.clustersize
    genplots = args.plots

    if args.seed is None:
        seed = np.random.randint(2**32, dtype=np.uint32)
    else:
        seed = int(args.seed, 16)
    np.random.seed(seed)
    seed_state = np.random.randint(2**32, dtype=np.uint32)
    seed_leak = np.random.randint(2**32, dtype=np.uint32)


    settings.precomp()

    #########################################
    #        HANDLE DEFAULT PARAMS          #
    #########################################

    if istate == '999':
        istate = 'RR' *((lane_length // 8) * 2) + '00' * ((lane_length // 8) * 23)
    if imask == '999':
        imask = '00' * 16 +  'FF'* (((lane_length // 8) * 25) - 16)
    if fmask == '999':
        fmask = '0'

    ####################################
    #        HARDCODED PARAMS          #
    ####################################

    entropy_threshold = 1.0 # abort when total entropy in graph < 1 bit
    change_threshold = 0.001 # abort when maximum statistical distance < 0.1%
    num_figs = 12

    ##############################
    #        PREPARE BP          #
    ##############################

    # print all used parameters

    print(f'name: {name}')
    print(f'lane_length: {lane_length}')
    print(f'rounds: {rounds}')
    print(f'istate: {istate}')
    print(f'imask: {imask}')
    print(f'fmask: {fmask}')
    print(f'word_size: {word_size}')
    print(f'sigma: {sigma}')
    print(f'iterations: {iterations}')
    print(f'name: {name}')
    print(f'damping: {settings.DAMP}')
    print(f'clustersize: {settings.CLUSTERSIZE}')
    print(f'seed: {seed:08X}')

    # prepare plot directory
    if genplots:
        plotpath = prepare_dir(name)

    # build graph

    keccak.setLaneLength(lane_length)
    computeConstants(lane_length)
    graph, t = build_graph(lane_length, rounds)

    print('Done building graph, now adding input')

    # conversion from hex string to keccak state
    np.random.seed(seed_state)
    initial_state = hex2keccak(istate, lane_length, int)

    initial_mask = hex2keccak(imask, lane_length, bool)
    final_mask = hex2keccak(fmask, lane_length, bool)

    # add leakage to graph
    chileak = not args.nochileak
    np.random.seed(seed_leak)
    add_leakage(graph, initial_state, initial_mask, final_mask, lane_length, rounds, word_size, sigma, chileak=chileak, noleak=args.noleak)

    print("Initializing graph and spreading priors")
    graph.initMessages()
    graph.spreadPriors()

    # prepare plots and plot prior entropy
    if genplots:
        figs = []
        for __ in range(num_figs):
            figs += [plt.figure()]

        iteration = 0
        for tx in range(3):
            fname = f"{plotpath}/entropy_t_{tx}_iteration_{iteration:02d}"
            print_state_property(graph, tx, fname=fname, fig=figs[tx])

        fname = f"{plotpath}/entropy_layers_iteration_{iteration:02d}"
        print_layer_property(graph, t, fname=fname, fig=figs[6])

        fname = f"{plotpath}/entropy_pplane_iteration_{iteration:02d}"
        print_plane_property(graph, "p", 0, fname=fname, fig=figs[8])
        fname = f"{plotpath}/entropy_tplane_iteration_{iteration:02d}"
        print_plane_property(graph, "t", 0, fname=fname, fig=figs[9])

    # calculate expected result

    expected_result = deepcopy(initial_state)
    for round_idx in range(rounds):
        expected_result = keccak.round(expected_result, round_idx)

    ##########################
    #        RUN BP          #
    ##########################

    abort_condition = "I"

    print("Starting BP")

    for itcount in range(iterations):
        iteration = itcount + 1
        print('Iteration: %d/%d' % (iteration, iterations))
        start = time.time()
        graph.scheduledUpdate()
        graph.norm()
        end = time.time()
        print('Iteration time: %f s' % (end - start))

        # check nodes at the input (much more info than on output)
        numerrors = 0

        for x in range(5):
            for y in range(5):
                for z in range(0, lane_length, settings.CLUSTERSIZE):
                    node = graph.getNode(nName(x, y, z, 0))
                    for zi in range(settings.CLUSTERSIZE):
                        bit = node.bits[zi]

                        if initial_state[x][y][z+zi] != bit:
                            # print(f"mismatch: node({x}, {y}, {z}), expected: {expectedResult[x][y][z]}, got: {bit}, dist: [{nodeDist[0]:.2f}, {nodeDist[1]:.2f}]")
                            numerrors += 1

        print('errors: %d' % (numerrors))

        if genplots:
            for tx in range(3):
                fname = f"{plotpath}/entropy_t_{tx}_iteration_{iteration:02d}"
                print_state_property(graph, tx, fname=fname, fig=figs[tx])
            for tx in range(3):
                fname = f"{plotpath}/change_t_{tx}_iteration_{iteration:02d}"
                print_state_property(graph, tx, fname=fname, fig=figs[tx + 3], property="change")

            fname = f"{plotpath}/entropy_layers_iteration_{iteration:02d}"
            print_layer_property(graph, t, fname=fname, fig=figs[6])
            fname = f"{plotpath}/change_layers_iteration_{iteration:02d}"
            print_layer_property(graph, t, fname=fname, fig=figs[7], prop="change")

            fname = f"{plotpath}/entropy_pplane_iteration_{iteration:02d}"
            print_plane_property(graph, "p", 0, fname=fname, fig=figs[8])
            fname = f"{plotpath}/entropy_tplane_iteration_{iteration:02d}"
            print_plane_property(graph, "t", 0, fname=fname, fig=figs[9])
            fname = f"{plotpath}/change_pplane_iteration_{iteration:02d}"
            print_plane_property(graph, "p", 0, fname=fname, fig=figs[10], property="change")
            fname = f"{plotpath}/change_tplane_iteration_{iteration:02d}"
            print_plane_property(graph, "t", 0, fname=fname, fig=figs[11], property="change")


        e = graph.nodesProperty(func=sum, property="entropy")
        print(f"Entropy sum tot: {e:.2f}")
        e0 = graph.sumEntropyT(0)
        print(f"Entropy sum t=0: {e0:.2f}")
        if e0 < entropy_threshold and settings.DAMP < 1.0:
            print("Entropy threshold in first layer reached, setting damp to 1")
            settings.DAMP = 1.0

        # if entropy of entire graph below some threshold --> abort
        if e < entropy_threshold:
            print(f"Entropy threshold, abort after iteration {iteration}")
            abort_condition = "E"
            break

        change = graph.nodesProperty(func=max, property="change")
        print(f"Change max: {change:.4f}")

        # if change below some threshold --> bp has converged
        if change < change_threshold:
            print(f"Change threshold, convergence after iteration {iteration}")
            abort_condition = "C"
            break

    #################################
    #        CHECK RESULTS          #
    #################################

    # check nodes at input and output
    numerrors_in = 0
    for x in range(5):
        for y in range(5):
            for z in range(0, lane_length, settings.CLUSTERSIZE):
                node = graph.getNode(nName(x, y, z, 0))
                for zi in range(settings.CLUSTERSIZE):
                    bit = node.bits[zi]
                    nodeDist = node.bitsDist[zi, :]

                    if initial_state[x][y][z+zi] != bit:
                        # print(f"in  mismatch: node({x}, {y}, {z+zi}), expected: {expected_result[x][y][z+zi]}, got: {bit}, dist: [{nodeDist[0]:.1f}, {nodeDist[1]:.1f}]")
                        numerrors_in += 1

    numerrors_out = 0
    for x in range(5):
        for y in range(5):
            for z in range(0, lane_length, settings.CLUSTERSIZE):
                node = graph.getNode(nName(x, y, z, t))
                for zi in range(settings.CLUSTERSIZE):
                    nodeDist = node.bitsDist[zi, :]
                    bit = node.bits[zi]

                    if expected_result[x][y][z+zi] != bit:
                        # print(f"out mismatch: node({x}, {y}, {z+zi}), expected: {expected_result[x][y][z+zi]}, got: {bit}, dist: [{nodeDist[0]:.1f}, {nodeDist[1]:.1f}]")
                        numerrors_out += 1

    print('errors at input : %d' % (numerrors_in))
    print('errors at output: %d' % (numerrors_out))

    e0 = graph.sumEntropyT(0)
    print(f"LOG {sigma}, {settings.DAMP}, {seed:08X}, {abort_condition}, {graph.iter}, {numerrors_in}, {numerrors_out}, {e0}")

    # return codes
    # 0: entropy (finished and found solution)
    # 1: assertion (can't change that)
    # 2: iterations
    # 3: convergence, but no success

    if (args.blocking):
        plt.show()  # if there are still plots open --> block

    if abort_condition == "E":
        sys.exit(0)
    elif abort_condition == "I":
        sys.exit(2)
    elif abort_condition == "C":
        sys.exit(3)

    sys.exit(-1) #what?

if __name__== "__main__":
    main()
