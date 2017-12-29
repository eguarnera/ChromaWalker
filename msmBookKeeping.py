#!/usr/bin/env python
"""
Collection of data bookkeeping functions and utilities for Hi-C analysis

Part of ChromaWalker package
"""


import numpy as np
import os
import sys
import cPickle as pickle
import hicFileIO as hfio
import dataFileReader as dfr


###########################################################

# Bookkeeping functions for reading MSM / TPT / target set data


def _get_tset_20160801(pars, tsetdict):
    """
    Get selected target set from single chromosome.
    """
    beta = pars.get('beta', 1.0)
    ntarget = pars['ntarget']
    key = (beta, ntarget)
    if key not in tsetdict:
        return []
    else:
        return tsetdict[key]


def _get_rho_20160801(pars, rhodict):
    """
    Get selected target set rho from single chromosome.
    """
    beta = pars.get('beta', 1.0)
    ntarget = pars['ntarget']
    key = (beta, ntarget)
    if key not in rhodict:
        return np.nan
    rho = rhodict[key]
    return rho


def _get_tsetReference_20180802(pars, ntarget):
    """
    Get reference target set for pars['chrfullname'].
    """
    tsetdatadir = pars['tsetdatadir']
    tsetdataprefix = pars['tsetdataprefix']
    region = pars.get('region', 'full')
    res = pars['res']
    resname = str(res / 1000) + 'kb'
    chrfullname = pars['chrfullname']
    tsetdataset = dfr._get_tsetdataset2(pars)
    dirname = os.path.join(tsetdatadir, tsetdataprefix,
                    chrfullname, region, resname)
    fname = os.path.join(dirname, tsetdataset +
                    '-reftsetdict_20160802.p')
    tsetdict = hfio._pickle_secureread(fname, free=True)
    if ntarget not in tsetdict:
        print 'ntarget=%i not in reference: Using default reference.' % ntarget
        return tsetdict['ref']
    else:
        return tsetdict[ntarget]


def _get_tsets_inter(pars, tsetdict):
    """
    Get selected target sets from chromosome pair.
    """
    region = pars.get('region', 'full')
    beta = pars.get('beta', 1.0)
    res = pars['res']
    chrfullname1 = pars['chrfullname1']
    chrfullname2 = pars['chrfullname2']
    ntarget1 = pars['ntarget1']
    ntarget2 = pars['ntarget2']
    tsetdataset = dfr._get_tsetdataset2(pars)
    key1 = (tsetdataset, chrfullname1, region, beta, res, ntarget1)
    key2 = (tsetdataset, chrfullname2, region, beta, res, ntarget2)
    tset1 = tsetdict[key1]
    tset2 = tsetdict[key2]
    return tset1, tset2


def _get_tsetmapkey(datadir, tset, norm='raw'):
    """
    Get unique identifier used to label target sets on the same Hi-C array.
    If tset is not currently mapped, give new ID.
    """
    # Get and lock tsetmap
    fname = os.path.join(datadir, 'tsetmap.p')
    tsetmap = hfio._pickle_secureread(fname, free=False)
    if tsetmap is None:
        freed = True
        tsetmap = {}
    else:
        freed = False
    tsetindex = None
    for index, (targetset, n) in tsetmap.iteritems():
        if len(targetset) != len(tset):
            continue
        if not np.allclose(targetset, tset) and n == norm:
            tsetindex = index
            break
    if tsetindex is None:
        print 'Compute effective Laplacian...'
        # Update dict
        tsetindex = len(tsetmap)
        tsetmap[tsetindex] = tset, norm
        hfio._pickle_securedump(fname, tsetmap, freed=freed)
    else:
        # Remove lock
        fname2 = fname + '-lock'
        os.remove(fname2)
    return tsetindex


#################################################

# To deprecate...


def _get_tset(pars, tsetdict):
    """
    Get selected target set from single chromosome.
    """
    region = pars.get('region', 'full')
    beta = pars.get('beta', 1.0)
    res = pars['res']
    chrfullname = pars['chrfullname']
    ntarget = pars['ntarget']
    tsetdataset = dfr._get_tsetdataset2(pars)
    key = (tsetdataset, chrfullname, region, beta, res, ntarget)
    if key not in tsetdict:
        return []
    else:
        return tsetdict[key]


def _get_rho(pars, rhodict):
    """
    Get selected target set rho from single chromosome.
    """
    region = pars.get('region', 'full')
    beta = pars.get('beta', 1.0)
    res = pars['res']
    chrfullname = pars['chrfullname']
    ntarget = pars['ntarget']
    tsetdataset = dfr._get_tsetdataset2(pars)
    key = (tsetdataset, chrfullname, region, beta, res, ntarget)
    if key not in rhodict:
        return np.nan
    rho = rhodict[key]
    return rho


def _get_tsetReference(pars, refdate='20160628'):
    """
    Get reference target set (based on standardized optimal ntarget)
    from single chromosome.
    """
    tsetdatadir = pars['tsetdatadir']
    region = pars.get('region', 'full')
    beta = pars.get('beta', 1.0)
    res = pars['res']
    chrfullname = pars['chrfullname']
    ntarget = pars['ntarget']
    tsetdataset = dfr._get_tsetdataset2(pars)
    key = (tsetdataset, chrfullname, region, beta, res, ntarget)
    fname = os.path.join(tsetdatadir, 'ReferenceTsets-%s.p' % refdate)
    if not os.path.isfile(fname):
        print 'Reference tset data file %s not found!' % fname
        sys.exit(1)
    tsetdict = pickle.load(open(fname, 'rb'))
    for k in tsetdict:
        print k
    tset = tsetdict[key]
    return tset

