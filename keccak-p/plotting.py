import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import glob

import keccak
import settings
from utils import nName, pName, fName


def print_state_property(G, t, fname=None, fig=None, property="bitsEntropy"):
    l = keccak.LANE_LENGTH
    c = settings.CLUSTERSIZE
    entmat = np.zeros((5, 5, l),dtype=settings.NUMPY_DATATYPE)
    for x in range(5):
        for y in range(5):
            for z in range(0, l, c):
                n = G.getNode(nName(x, y, z, t))
                entmat[x, y, z:(z+c)] = getattr(n, property)

    if not fig:
        fig = plt.gcf()
    plt.figure(fig.number)
    fig.clf()

    cmap = (plt.cm.get_cmap('hot')).reversed()
    ax = []
    for y in range(5):
        currax = fig.add_subplot(5, 1, y+1)
        ax += [currax]
        e = np.squeeze(entmat[:, y, :])
        im = currax.imshow(e, cmap=cmap, vmin=0, vmax=1)
        plt.yticks([])
        plt.xticks([])
        plt.ylabel(f"y={y}")

    fig.colorbar(im, ax=ax)

    plt.pause(0.001)
    if fname:
        fig.savefig(fname)


def print_plane_property(G, y, r, fname=None, fig=None, property="bitsEntropy"):
    l = keccak.LANE_LENGTH
    c = settings.CLUSTERSIZE
    entmat = np.zeros((5, l),dtype=settings.NUMPY_DATATYPE)
    for x in range(5):
        for z in range(0, l, c):
            n = G.getNode(nName(x, y, z, r))
            entmat[x, z:(z+c)] = getattr(n, property)

    if not fig:
        fig = plt.gcf()
    plt.figure(fig.number)
    fig.clf()

    cmap = (plt.cm.get_cmap('hot')).reversed()
    plt.imshow(entmat, cmap=cmap, vmin=0, vmax=1)
    plt.yticks([])
    plt.xticks([])

    plt.colorbar()

    plt.pause(0.001)
    if fname:
        fig.savefig(fname)


def print_layer_property(G, tmax, fname=None, fig=None, prop="bitsEntropy"):
    if not fig:
        fig = plt.gcf()
    plt.figure(fig.number)
    plt.clf()

    entvec = np.zeros(tmax+1, dtype=settings.NUMPY_DATATYPE)
    stdvec = np.zeros(tmax+1, dtype=settings.NUMPY_DATATYPE)
    for t in range(tmax+1):
        nodes = G.filterNodesT(t)
        hvec = np.array([getattr(a, prop) for a in nodes])
        entvec[t] = np.mean(hvec)
        stdvec[t] = np.std(hvec)

    plt.plot(range(tmax+1), entvec)
    plt.xlabel("t")
    plt.ylabel("Avg. h")
    plt.gca().set_ylim([0,1])

    plt.pause(0.001)
    if fname:
        fig.savefig(fname)


def prepare_dir(name, ending="png"):
    try:
        os.mkdir('plots')
    except FileExistsError:
        pass
    path = 'plots' + os.sep + name
    try:
        os.mkdir(path)
        print("Directory", path, "created")
    except FileExistsError:
        print("Directory", path, "already exists")
        filelist = glob.glob(f"{path}/*.{ending}", recursive=False)
        for f in filelist:
            os.remove(f)
        print("Directory", path, "cleared")
    return path
