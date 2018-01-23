#!/usr/bin/env python
"""
Miscellaneous functions for Hi-C graph data analysis

Part of ChromaWalker package
"""

import numpy as np
from scipy.stats import gmean
import pandas as pd


#####################

# Epigenetics data handling

def _epigen_extractDensity(track, partitiondata, mode='lin'):
    """
    Extract partition-average density values from epigen data track.
    Parameter mode: 'lin' for arithmetic mean, 'log' for geometric mean.
    """
    if mode == 'lin':
        func = np.average
    elif mode == 'log':
        func = gmean
    else:
        print '_epigen_extractDensity: Invalid mode %s!' % mode
        return
    data = {}
    for key, (st, en) in partitiondata:
        data[key] = func(track[st:en])
    return data


def _epigen_vecToZScore(vec):
    """
    Convert vector of values into Z scores.
    """
    mean = np.average(vec)
    sd = np.std(vec) * np.sqrt(len(vec) / (len(vec) - 1.0))
    return (vec - mean) / sd


def _epigen_vecToZScore_weighted(vec, weights, printsummary=False):
    """
    Convert vector of values (weighted) into Z scores.
    """
    mean = np.sum(vec * weights) / np.sum(weights)
    sd = np.sum((vec ** 2 - mean ** 2) * weights) / np.sum(weights) * \
            (len(vec) / (len(vec) - 1.0))
    sd = np.sqrt(sd)
    if printsummary:
        print 'minval maxval meanval sigma sigma/mu minZ maxZ = ' + \
            '\t%e\t%e\t%e\t%e\t%e\t%e\t%e' % (np.min(vec), np.max(vec), mean, sd,
                sd / mean, np.min((vec - mean) / sd), np.max((vec - mean) / sd))
    return (vec - mean) / sd


def _epigen_dumpNodeData(datalist, cols, fname, sep=','):
    """
    Write datalist[rowindex][colindex] to file fname, with separator sep.
    Each datalist[rowindex] is a list of quantities corresponding to cols.
    """
    datadict = {col: datalist[icol] for icol, col in enumerate(cols)}
    df = pd.DataFrame(datadict, columns=cols)
    df.to_csv(fname, sep=sep, index=False)
    return
