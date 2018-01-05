#!/usr/bin/env python
"""
Collection of basic clustering functions and utilities for Hi-C analysis

Part of ChromaWalker package
"""

import numpy as np
import os
import subprocess
import sys
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import copy
import matplotlib.gridspec as gridspec
from time import time, sleep
from numpy.linalg import solve
import cPickle as pickle
import matplotlib.backends.backend_pdf
from matplotlib import cm
from matplotlib import collections as mc
#import mpldatacursor
import plotutils as plu
from scipy.stats import norm as statnorm
from scikits.bootstrap import bootstrap
from numpy.linalg import eigvals as eigvalsnp
from operator import itemgetter
import gzip
from Bio import SeqIO
from Bio.SeqUtils import GC


## Basic clustering operations


def _clusterproximity_00(set1, set2, ftot, sizes):
    """
    Compute proximity of 2 clusters, by size-weighted interaction counts.
    For basic clustering analysis only.
    Input:
        ftot: Array of total interactions between units
        sizes: Sizes of units
    Returns:
        favgval: Size-weighted (averaged) interaction between the two sets
                 set1, set2
    """
    s1 = np.array(set1)
    s2 = np.array(set2)
    ntot = np.sum(ftot[s1][:, s2])
    favgval = ntot / (np.sum(sizes[s1]) * np.sum(sizes[s2]))
    return favgval


def _clusterproximity_01(set1, set2, ftot, sizes):
    """
    Compute proximity of 2 clusters, by minimum proximity
    between constituent units.
    For basic clustering analysis only.
    Input:
        ftot: Array of total interactions between units
        sizes: Sizes of units (not used)
    Returns:
        fminval: Minimum interaction between units of the two sets
                 set1, set2
    """
    s1 = np.array(set1)
    s2 = np.array(set2)
    fminval = np.min(ftot[s1][:, s2])
    return fminval


def _clustersplit_minprox(favg, cluster, hubs):
    """
    Compute min proximity to hubs for points in cluster.
    Uses array favg as measure of proximity between individual units.
    Note: hubs must only contain 2 points, and be a subset of cluster!
    Input:
        favg: Average interaction levels between units
        cluster: Cluster of units to split
        hubs: Proposed hubs for splitting cluster
    Returns:
        minprox: Min proximity from cluster units to hubs
        splitting: (2,) list of indices closer to each hub
                   Note: Ties a awarded to the first hub
    """
    # Check hubs
    if len(hubs) != 2:
        print 'Incorrect number of hubs: %i!' % len(hubs)
        return
    if not set(hubs).issubset(set(cluster)):
        print 'hubs is not a subset of cluster!'
        return
    # Compute minprox
    cl = np.array(cluster)
    hb = np.array(hubs)
    favgsub = favg[cl][:, hb]
    minprox = np.min(favgsub)
    # Find splitting
    sub1 = list(cl[favgsub[:, 0] <= favgsub[:, 1]])
    sub2 = list(cl[favgsub[:, 0] > favgsub[:, 1]])
    return minprox, (sub1, sub2)
