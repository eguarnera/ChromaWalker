#!/usr/bin/env python
"""
Collection of miscellaneous analysis functions and utilities for Hi-C analysis

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


# Miscellaneous specialized analysis functions


## Average genomic distance between genomic loops,
##  as function of cytoband stain
### Single chromosome


def AverageLoopSize_1chr(pars, clrfind):
    """
    Find average genomic distances of interaction loops in Hi-C data.
    Consider intra-band contacts only, grouped by stain level.
    Gets data over single chromosome.
    Input:
        pars: Dict of run parameters.
              Note: Will force pars['chrref'] = pars['chrfullname']
        clrfind: List of stain labels to consider
    Returns:
        loopsizeavg: List of average contact distances for each stain label.
                     Note: For stains that do not exist in the chromosome
                           of choice, nan is returned.
    """
    ## Forced requirenment
    pars['chrref'] = copy.deepcopy(pars['chrfullname'])
    ## Pars parameters
    norm = pars['norm']
    res = pars['res']
    ## Read fmat, cytobanddata, mappable chromsizes
    fmat, _, _, mappingdata = _get_arrays(_get_runbinarydir(pars),
                    norm=norm)
    mapping = mappingdata[0]
    cytoband, blimlist, bnamelist, clrlist = _get_cytoband(pars)
    ## Compute average distances
    nclrs = len(clrfind)
    bigsize = np.max(np.array(blimlist)[:, 1] - np.array(blimlist)[:, 0]) + 1
    weights = np.zeros((nclrs, int(np.ceil(bigsize))))
    print bigsize
    for blims, clr in zip(blimlist, clrlist):
        if clr not in clrfind:
            continue
        clrind = clrfind.index(clr)
        ### Find which fmat positions are in this band
        thisband = []
        a, b = map(int, map(np.ceil, blims))
        for px in range(a, b + 1):
            if px in mapping:
                thisband.append(list(mapping).index(px))
        thisband = np.array(thisband)
        ### Extract weights
        for p1 in thisband:
            for p2 in thisband:
                if p2 >= p1:
                    continue
                weights[clrind, mapping[p1] - mapping[p2]] += fmat[p1, p2]
    ### Compute averages
    loopsizeavg = np.zeros(nclrs)
    for clrind in range(nclrs):
        loopsizeavg[clrind] = \
                    np.sum(weights[clrind] * np.arange(bigsize) * res / 1.0e6) \
                    / np.sum(weights[clrind])
    return loopsizeavg


### All chromosomes

def AverageLoopSize_Nchr(pars, clrfind, chrfullnamelist):
    """
    Find average genomic distances of interaction loops in Hi-C data.
    Consider intra-band contacts only, grouped by stain level.
    Gets data over list of chromosomes.
    Input:
        pars: Dict of run parameters.
              Note: Will force pars['chrref'] = pars['chrfullname']
        clrfind: (M,) List of stain labels to consider
        chrfullnamelist: (N,) List of chromosome names: 'chr1', etc.
    Returns:
        loopsizeavg: (M, N) Array of average contact distances
                     for each stain label.
                     Note: For stains that do not exist in the chromosome
                           chosen, nan is returned.
    """
    parstemp = copy.deepcopy(pars)
    lsalist = []
    for chrfullname in chrfullnamelist:
        parstemp['chrfullname'] = chrfullname
        parstemp['chrref'] = chrfullname
        lsalist.append(AverageLoopSize_1chr(parstemp, clrfind))
    return np.array(lsalist).T


## Band-wise interaction
### Single chromosome

def BinnedInteractions_Band_1chr(pars):
    """
    Create reduced matrix of binned interactions.
    Band-wise binning: Use cytobands as partitioning.
    Input:
        pars['bandbin']: 'average' or 'sum'... Should be self-evident.
    Returns:
        fb1b2: (M, M) array of inter-band interactions
        fb1b2ext1: (N1, N1) extended array of interactions, same dimensions as
                   fmat.
        fb1b2ext2: (N2, N2) extended array of interactions, same dimensions as
                   fmatpad.
    """
    bandbin = pars['bandbin']
    norm = pars['norm']
    if bandbin == 'average':
        binfun = np.average
    elif bandbin == 'sum':
        binfun = np.sum
    else:
        print 'Invalid parameter bandbin: %s!' % bandbin
        sys.exit(1)
    # Read fmat
    fmat, _, _, mappingdata = _get_arrays(_get_runbinarydir(pars),
                    norm=norm)
    mapping, nbins = mappingdata
    # Read cytoband
    cytoband, blimlist, bnamelist, clrlist = _get_cytoband(pars)
    # Create bandnum array
    bandnum0 = np.zeros(nbins, dtype=int)
    for i, blims in enumerate(blimlist):
        ia, ib = map(int, blims)
        bandnum0[ia:ib + 1] = i
    bandnum = np.zeros(len(fmat), dtype=int)
    for i in range(len(fmat)):
        bandnum[i] = bandnum0[mapping[i]]
    # Bin bands, excluding 'acen' only
    nbands = len(clrlist)
    fb1b2 = np.zeros((nbands, nbands))
    for ib1, cl1 in enumerate(clrlist):
        if cl1 == 'acen':
            continue
        mask1 = (bandnum == ib1)
        for ib2, cl2 in enumerate(clrlist):
            if cl2 == 'acen':
                continue
            mask2 = (bandnum == ib2)
            fb1b2[ib1, ib2] = binfun(fmat[mask1][:, mask2])
    fb1b2ext1 = plu._extendarray(fb1b2, bandnum, membermode='hard')
    fb1b2ext2 = plu._extendarray(fb1b2, bandnum0, membermode='hard')
    return fb1b2, fb1b2ext1, fb1b2ext2


## Target density as function of cytoband stain
### All chromosomes


def _bootstrap_chrbandstains(bandvals, alpha=0.05, nsamples=20000):
    """
    Create bootstrap confidence intervals for average of bandvals,
    across all chromosomes (retaining number of samples of each stain level
    per chr), across stain levels.
    Input:
        bandvals: (M, Nchrs, N(M)) List of values to average over.
                  M = number of stain levels
                  Nchrs = number of chromosomes
                  N(M) = number of bands of given stain level in given chr
        bstrintervals: (M, 2) alpha- confidence intervals of mean bandvals.
    """
    nclrs = len(bandvals)
    gbands = [[] for g in range(nclrs)]
    for i in range(nclrs):
        for a in bandvals[i]:
            gbands[i] = gbands[i] + list(a)
    gavgs = map(np.average, gbands)
    gbandsarr = [np.array(gb) for gb in gbands]
    alphas = np.array([alpha / 2, 1 - alpha / 2])
    # Bias correction
    gz0s = [statnorm.ppf((1.0 * np.sum(gbandsarr[i] < gavgs[i], axis=0)) /
                    len(gbandsarr[i])) for i in range(nclrs)]
    # Jackknife statistics
    gas = []
    for i in range(nclrs):
        jackindexes = bootstrap.jackknife_indexes(gbandsarr[i])
        jstat = [np.average(gbandsarr[i][indexes]) for indexes in jackindexes]
        jmean = np.mean(jstat, axis=0)
        gas.append(np.sum((jmean - jstat) ** 3, axis=0) /
                        (6.0 * np.sum((jmean - jstat) ** 2, axis=0) ** 1.5))
    # z values, quantile positions
    gzss = [gz0s[i] + statnorm.ppf(alphas).reshape(alphas.shape + (1,) *
                    gz0s[i].ndim)
                    for i in range(nclrs)]
    gavalss = [statnorm.cdf(gz0s[i] + gzss[i] / (1 - gas[i] * gzss[i]))
                    for i in range(nclrs)]
    gnvalss = [np.round((nsamples - 1) * gavalss[i]).astype('int')
                    for i in range(nclrs)]
    # Resample
    gnbandschr = [map(len, bandvals[i])
                    for i in range(nclrs)]
    gsampless = [[] for i in range(nclrs)]
    for i in range(nsamples):
        for j in range(nclrs):
            thisclrsample = []
            for k, nbands in enumerate(gnbandschr[j]):
                if nbands == 0:
                    continue
                thisclrsample = thisclrsample + list(bandvals[j][k][
                                    np.random.randint(0, nbands, nbands)])
            gsampless[j].append(np.average(thisclrsample))
    gsampless2 = [np.sort(gs) for gs in gsampless]
    bstrintervals = [np.array(gsampless2[i])[gnvalss[i]] for i in range(nclrs)]
    return np.array(bstrintervals)


def AverageTargetDensity_Nchr(pars, clrfind, chrfullnamelist,
                bootstrappars=None):
    """
    Compute average target density as a function of stain level, across
    list of chromosomes. For each chromosome, ntarget is chosen to match a mean
    target density.
    Input:
        pars['meanpartsize']: Average distance between targets.
        clrfind: List of stain levels to look at.
        chrfullnamelist: Self-evident...
        bootstrappars: A list of values to determine whether or not to
                       compute bootstrap confidence intervals, and parameters.
    Returns:
        tgtdens: (M, N) array of average target density.
                 Note: M = number of stain levels
                       N = number of chromosomes
        bandtgtdens:(M, N, Nband(M)) list of target density in each band.
                    Note: Nband(M) = number of bands of given stain
                          in given chromosome.
    """
    # Unpack pars
    meanpartsize = pars['meanpartsize']
    tsetdatadir, tsetdataprefix = pars['tsetdatadir'], pars['tsetdataprefix']
    norm = pars['norm']
    region = pars.get('region', 'full')
    beta = pars['beta']
    res = pars['res']
    if bootstrappars is None or not bootstrappars[0]:
        bootstrapon = False
    else:
        bootstrapon, alpha, nsamples = bootstrappars
    _, tsetdict = _get_rhotsetdicts(tsetdatadir, tsetdataprefix)
    nclrs = len(clrfind)
    nchrs = len(chrfullnamelist)
    tgtdens = np.zeros((nclrs, nchrs))
    bandtgtdens = [[0 for chrfullname in chrfullnamelist] for g in clrfind]
    chrsizes = _get_allchrmappedsizes(pars, chrfullnamelist) * res / 1.0e6
    for chrid, chrfullname in enumerate(chrfullnamelist):
        pars['chrfullname'] = chrfullname
        pars['chrref'] = chrfullname
        mappingdata = _get_mappingdata(_get_runbinarydir(pars),
                        norm=norm)
        mapping = mappingdata[0]
        cytoband, blimlist, bnamelist, clrlist = _get_cytoband(pars)
        ntarget = int(np.ceil(chrsizes[chrid] / meanpartsize))
        key = (_get_tsetdataset2(pars), chrfullname, region,
                        beta, res, ntarget)
        tset = tsetdict[key]
        ## Compute target counts
        nbands = len(blimlist)
        tcounts = np.zeros(nclrs)
        sizes = np.zeros(nclrs)
        bandtcounts = np.zeros(nbands)
        bandsizes = np.zeros(nbands)
        for i, ((st, en), clr) in enumerate(zip(blimlist, clrlist)):
            if clr not in clrfind:
                continue
            sizes[clrfind.index(clr)] += (en - st) * res / 1.0e6
            bandsizes[i] = (en - st) * res / 1.0e6
        for t in tset:
            pos = mapping[t]
            for i, ((st, en), clr) in enumerate(zip(blimlist, clrlist)):
                if clr not in clrfind:
                    continue
                if pos >= st and pos < en:
                    tcounts[clrfind.index(clr)] += 1
                    bandtcounts[i] += 1
        ## Compute target density
        for i, g in enumerate(clrfind):
            clrmask = (np.array(clrlist) == g)
            tgtdens[i, chrid] = tcounts[i] / sizes[i]
            bandtgtdens[i][chrid] = bandtcounts[clrmask] / bandsizes[clrmask]
    if bootstrapon:
        print 'Bootstrapping...'
        bstrintervals = _bootstrap_chrbandstains(bandtgtdens)
        return tgtdens, bandtgtdens, np.array(bstrintervals)
    else:
        return tgtdens, bandtgtdens


def AveragePartSize_Nchr(pars, clrfind, chrfullnamelist, alllevels,
                bootstrappars=None):
    """
    Compute average target density as a function of stain level, across
    list of chromosomes. For each chromosome, ntarget is chosen to match a mean
    target density.
    Input:
        pars['meanpartsize']: Average distance between targets.
        clrfind: List of stain levels to look at.
        chrfullnamelist: Self-evident...
        bootstrappars: A list of values to determine whether or not to
                       compute bootstrap confidence intervals, and parameters.
    Returns:
        tgtdens: (M, N) array of average target density.
                 Note: M = number of stain levels
                       N = number of chromosomes
        bandtgtdens:(M, N, Nband(M)) list of target density in each band.
                    Note: Nband(M) = number of bands of given stain
                          in given chromosome.
    """
    # Unpack pars
    meanpartsize = pars['meanpartsize']
    tsetdatadir, tsetdataprefix = pars['tsetdatadir'], pars['tsetdataprefix']
    norm = pars['norm']
    region = pars.get('region', 'full')
    beta = pars['beta']
    res = pars['res']
    if bootstrappars is None or not bootstrappars[0]:
        bootstrapon = False
    else:
        bootstrapon, alpha, nsamples = bootstrappars
    _, tsetdict = _get_rhotsetdicts(tsetdatadir, tsetdataprefix)
    nclrs = len(clrfind)
    nchrs = len(chrfullnamelist)
    partsize = np.zeros((nclrs, nchrs))
    bandpartsize = [[0 for chrfullname in chrfullnamelist] for g in clrfind]
    chrsizes = _get_allchrmappedsizes(pars, chrfullnamelist) * res / 1.0e6
    for chrid, chrfullname in enumerate(chrfullnamelist):
        pars['chrfullname'] = chrfullname
        pars['chrref'] = chrfullname
        beta,
        mappingdata = _get_mappingdata(_get_runbinarydir(pars),
                        norm=norm)
        mapping = mappingdata[0]
        cytoband, blimlist, bnamelist, clrlist = _get_cytoband(pars)
        ntarget = int(np.ceil(chrsizes[chrid] / meanpartsize))
        key = (_get_tsetdataset2(pars), chrfullname, region,
                        beta, res, ntarget)
        tset = tsetdict[key]
        ## Compute target counts
        nbands = len(blimlist)
        tcounts = np.zeros(nclrs)
        sizes = np.zeros(nclrs)
        bandtcounts = np.zeros(nbands)
        bandsizes = np.zeros(nbands)
        for i, ((st, en), clr) in enumerate(zip(blimlist, clrlist)):
            if clr not in clrfind:
                continue
            sizes[clrfind.index(clr)] += (en - st) * res / 1.0e6
            bandsizes[i] = (en - st) * res / 1.0e6
        for t in tset:
            pos = mapping[t]
            for i, ((st, en), clr) in enumerate(zip(blimlist, clrlist)):
                if clr not in clrfind:
                    continue
                if pos >= st and pos < en:
                    tcounts[clrfind.index(clr)] += 1
                    bandtcounts[i] += 1
        ## Compute target density
        for i, g in enumerate(clrfind):
            clrmask = (np.array(clrlist) == g)
            partsize[i, chrid] = tcounts[i] / sizes[i]
            bandpartsize[i][chrid] = bandtcounts[clrmask] / bandsizes[clrmask]
    if bootstrapon:
        print 'Bootstrapping...'
        bstrintervals = _bootstrap_chrbandstains(bandpartsize, alpha=alpha,
                        nsamples=nsamples)
        return partsize, bandpartsize, np.array(bstrintervals)
    else:
        return partsize, bandpartsize