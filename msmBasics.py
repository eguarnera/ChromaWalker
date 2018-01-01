#!/usr/bin/env python
"""
Collection of basic MSM computation functions and utilities for Hi-C analysis

Part of ChromaWalker package
"""


import numpy as np
from numpy.linalg import solve
from numpy.linalg import eigvals as eigvalsnp


## Basic graph / MSM computations


def _LEMAnalysis(data):
    """
    Basic Laplacian eigenmap analysis: Identify structural hierarchies.

    Returns 3 lists: eratios, ivals, evals
        - evals = eigenvalues of L sorted in increasing order
        - eratios = evals[i] / evals[i+1] for i in range(1, N)
        - ivals = range(1, N)
    """
    nbins = data.shape[0]
    pmat = data.copy()
    for i in range(nbins):
        pmat[i] /= np.sum(pmat[i])
    lmat = pmat - np.eye(nbins)
    evals = eigvalsnp(lmat.T)
    evals.sort()
    evals = evals[::-1]
    eratios = evals[1:-1] / evals[2:]
    ivals = np.arange(len(eratios)) + 1
    return eratios, ivals, evals


def _calc_MFPT(fmat, loops=False):
    """
    Calculate Markov mean first pass time on a graph with
        vertex weight matrix fmat.
    If loops=False, ignore self-loops.
    """
    nbins = fmat.shape[0]
    # Markov transition probability
    if loops:
        pmat = fmat.copy()
    else:
        pmat = fmat - np.diag(np.diag(fmat))
    for i in range(nbins):
        pmat[i] = pmat[i] / np.sum(pmat[i])
    # Mean first-pass times
    mmat = np.zeros_like(pmat)
    ## Loop across columns
    for j in range(nbins):
        ## Temp pmat
        pmatt = pmat.copy()
        pmatt[:, j] = 0.0
        mmat[:, j] = solve(pmatt - np.eye(nbins), -np.ones(nbins))
    return mmat


def _calc_MFPT_withLoops(fmat):
    """
    Calculate Markov mean first pass time on a graph with
        vertex weight matrix fmat.
    """
    nbins = fmat.shape[0]
    # Markov transition probability
    pmat = fmat.copy()
    for i in range(nbins):
        pmat[i] = pmat[i] / np.sum(pmat[i])
    # Mean first-pass times
    mmat = np.zeros_like(pmat)
    ## Loop across columns
    for j in range(nbins):
        ## Temp pmat
        pmatt = pmat.copy()
        pmatt[:, j] = 0.0
        mmat[:, j] = solve(pmatt - np.eye(nbins), -np.ones(nbins))
    return mmat


def _calc_cmat(mmat):
    """
    Calculate committor from MFPT.
    """
    cmat = np.zeros_like(mmat)
    nbins = len(mmat)
    for i in range(nbins):
        for j in range(nbins):
            cmat[i, j] = mmat[i, i] / (mmat[i, j] +
                            mmat[j, i])
    return cmat - np.diag(np.diag(cmat))


####################################################

# To deprecate...


def _calc_MFPT_20160831(fmat, mapping):
    """
    Calculate Markov mean first pass time.
    """
    nbins = fmat.shape[0]
    # Markov transition probability
    pmat = fmat - np.diag(np.diag(fmat))
    for i in range(nbins):
        pmat[i] = pmat[i] / np.sum(pmat[i])
    # Mean first-pass times
    mmat = np.zeros_like(pmat)
    ## Loop across columns
    badloci = []
    for j in range(nbins):
        ## Temp pmat
        pmatt = pmat.copy()
        pmatt[:, j] = 0.0
        mmat[:, j] = solve(pmatt - np.eye(nbins), -np.ones(nbins))
        if np.sum(mmat[:, j] < 0.0) > 0:
            badloci.append(j)
    if len(badloci) > 0:
        # Modify fmat, mapping, mmat
        badloci.sort()
        badloci.reverse()
        mp = list(mapping)
        for i in badloci:
            del(mp[i])
        mapping = np.array(mp)
        goodinds = list(set(range(len(mmat))) - set(badloci))
        goodinds.sort()
        goodinds = np.array(goodinds)
        if len(goodinds) == 0:
            return None
        fmat = fmat[goodinds][:, goodinds]
        mmat = mmat[goodinds][:, goodinds]
    return fmat, mmat, mapping


