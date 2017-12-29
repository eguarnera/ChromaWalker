# Collection of analysis functions and utilities
# Specialized for Hi-C analysis

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


##########################################################################
# Data / archive utilities
## Basic file I/O handling


def _pickle_secureread(fname, free=False):
    """
    Read a pickled data structure securely (ensure only 1 thread reading it).
    free: Whether or not to let other processes read the same file after I'm
          done... Set to False to avoid race conditions.
    """
    if not os.path.isfile(fname):
        print '%s is not a file!' % fname
        return
    fname2 = fname + '-lock'
    while True:
        if not os.path.isfile(fname2):
            open(fname2, 'a').close()
            break
        print 'Waiting for file lock to free up...'
        sleep(1)
    data = pickle.load(open(fname, 'rb'))
    if free:
        os.remove(fname2)
    return data


def _pickle_securereads(fnames, free=False):
    """
    Read a pickled data structure securely (ensure only 1 thread reading it).
    free: Whether or not to let other processes read the same file after I'm
          done... Set to False to avoid race conditions.
    Reads list of files, with first file as lock.
    """
    #print 'Read', fnames
    fname = fnames[0]
    if not os.path.isfile(fname):
        print '%s is not a file!' % fname
        return
    fname2 = fname + '-lock'
    while True:
        if not os.path.isfile(fname2):
            open(fname2, 'a').close()
            break
        print 'Waiting for file lock to free up...'
        sleep(1)
    data = []
    for fn in fnames:
        data.append(pickle.load(open(fn, 'rb')))
    if free:
        os.remove(fname2)
    return data


def _pickle_securedump(fname, data, freed=False):
    """
    Write a pickled data structure securely (ensure only 1 thread writing it).
    freed: Whether or not the file access was locked. Set to False to prevent
           race conditions.
    """
    fname2 = fname + '-lock'
    if freed:
        while True:
            if not os.path.isfile(fname2):
                open(fname2, 'a').close()
                break
            print 'Waiting for file lock to free up...'
            sleep(1)
    pickle.dump(data, open(fname, 'wb'))
    os.remove(fname2)
    return


def _pickle_securedumps(fnames, datas, freed=False):
    """
    Write a pickled data structure securely (ensure only 1 thread writing it).
    freed: Whether or not the file access was locked. Set to False to prevent
           race conditions.
    Writes list of files, with first file as lock.
    """
    #print 'Write', fnames
    fname = fnames[0]
    fname2 = fname + '-lock'
    if freed:
        while True:
            if not os.path.isfile(fname2):
                open(fname2, 'a').close()
                break
            print 'Waiting for file lock to free up...'
            sleep(1)
    for fn, dt in zip(fnames, datas):
        pickle.dump(dt, open(fn, 'wb'))
    os.remove(fname2)
    return

## Reading data files: ASCII files


def _get_allchrsizes(chrnames, genomeref, genomedatadir):
    """
    Find out how many pixels are required to represent
    given chromosome at given resolution.
    """
    datavals = []
    for chrname in chrnames:
        chrfullname = chrname if chrname[:3] == 'chr' else 'chr' + chrname
        fname = os.path.join(genomedatadir,
                genomeref, genomeref + '.chrom.sizes')
        if not os.path.isfile(fname):
            print 'chrom.sizes file doesn\'t exist at', fname, '!'
            return
        with open(fname, 'r') as f:
            for line in f:
                data = line.split()
                if data[0] == chrfullname:
                    chrlength = float(data[1])
        datavals.append(chrlength)
    return datavals


def _get_cytoband(pars):
    """
    Read cytoband data and create array representation at given resolution.
    """
    genomedatadir = pars['genomedatadir']
    genomeref = pars['genomeref']
    chrname = pars['chrref']
    resolution = pars['res']
    # Define color dictionary
    colordict = {
        'gneg': 0.0,
        'gpos25': 1.0,
        'gpos50': 2.0,
        'gpos75': 3.0,
        'gpos100': 4.0,
        'acen': -1.0,
        'gvar': 0.5,
        'stalk': 0.25
        }
    # Define file name
    fname = os.path.join(genomedatadir, genomeref, 'cytoBand.txt')
    if not os.path.isfile(fname):
        print 'Cytoband file doesn\'t exist at', fname, '!'
        return
    # Read and store relevant lines
    data = []
    chrfullname = chrname if chrname[:3] == 'chr' else 'chr' + chrname
    with open(fname, 'r') as f:
        for line in f:
            if line.split()[0] == chrfullname:
                data.append(line)
    # Calculate array length
    arraylen = int(np.ceil(float(data[-1].split()[2]) / resolution)) + 1
    cytoband = np.zeros(arraylen)
    blimlist = []
    bnamelist = []
    clrlist = []
    # Fill in cytoband array data
    for line in data:
        _, st, en, bandname, clr = line.split()
        stpos = float(st) / resolution
        enpos = float(en) / resolution
        cytoband[int(stpos):int(enpos)] = colordict[clr]
        blimlist.append([stpos, enpos])
        bnamelist.append(bandname)
        clrlist.append(clr)
    return cytoband, blimlist, bnamelist, clrlist


def _get_blims(pars, bnamelist, bnamemap):
    """
    Get band boundaries, given list of band names in 'h3-1' format,
    and mapping to cytoband format.
    Note: bnamemap must correspond to the chromosome selected in pars.
    """
    cytoband, blimlist2, bnamelist2, clrlist = _get_cytoband(pars)
    blimlist = []
    for bname in bnamelist:
        ind = bnamelist2.index(bnamemap[bname])
        if ind < 0:
            print 'Band not found!'
            return None
        blim = map(int, np.array(blimlist2[ind]) * pars['res'])
        blimlist.append(blim)
    return blimlist


## Extract Hi-C data from scratch


def _calc_subsetmappingdata(selection, baseres, res):
    """
    Calculate data required to extract subregion interaction matrix
    from sparse-format file.
    """
    mapping = []
    # Calculate number of pixels  and pixel mapping for each interval
    for a, b in selection:
        # Start pixel
        st = int(np.floor(a / res)) * res
        # End pixel
        if b % int(res) == 0:
            #en = int(np.floor(b / res) - 1) * res
            en = int(np.floor(b / res)) * res
        else:
            en = int(np.floor(b / res)) * res
        thispixelmap = range(st / res, en / res + 1)
        mapping = mapping + thispixelmap
    #print mapping[-1]
    # Calculate data selection, weights, mapping to pixels, and pixel weights
    sliceselections = []
    sliceweights = []
    slicetopixels = []
    pixelweights = np.zeros_like(mapping) * 1.0
    for a, b in selection:
        st = int(np.floor(a / baseres)) * baseres
        if b % int(baseres) == 0:
            en = int(np.floor(b / baseres) - 1) * baseres
        else:
            en = int(np.floor(b / baseres)) * baseres
        if en - st < 1:
            print 'Warning: Slice [%i, %i] too small! Ignoring...' % (a, b)
            continue
        # Calculate selection and weights associated with each entry at baseres
        thissliceselection = range(st, en + baseres, baseres)
        thissliceweights = [(a - st) / (1.0 * baseres)] + \
                           [1.0 for i in range(st + baseres, en, baseres)] + \
                           [(b - en) / (1.0 * baseres)]
        # Save info on slice data selection
        sliceselections = sliceselections + thissliceselection
        sliceweights = sliceweights + thissliceweights
        # Calculate mapping from slice data selections to pixels
        thisslice2pixmap = [mapping.index(val / res)
                        for val in thissliceselection]
        slicetopixels = slicetopixels + thisslice2pixmap
        # Calculate total pixel weights
        for i, val in enumerate(thissliceselection):
            pixelweights[mapping.index(val / res)] += \
                            thissliceweights[i]
    return sliceselections, sliceweights, slicetopixels, pixelweights, mapping


def _get_chrdatasize(chrname, genomeref, genomedatadir, resolution):
    """
    Find out how many pixels are required to represent
    given chromosome at given resolution.
    """
    # Full chromosome name
    chrfullname = chrname if chrname[:3] == 'chr' else 'chr' + chrname
    fname = os.path.join(genomedatadir, genomeref, genomeref + '.chrom.sizes')
    if not os.path.isfile(fname):
        print 'chrom.sizes file doesn\'t exist at', fname, '!'
        return
    with open(fname, 'r') as f:
        for line in f:
            data = line.split()
            if data[0] == chrfullname:
                chrlength = int(data[1])
    nbins = int(np.ceil(chrlength * 1.0 / resolution))
    return nbins


def _extract_fij_subregion_LiebermannAiden2014(chrname,
                    genomeref, genomedatadir, hicfile,
                    baseres, baseresname, res, regionselection, nloop=0,
                    weightboundaries=False, norm='raw'):
    """
    Extract interaction matrix for subregion defined by selection.
    """
    # Find data size nbins
    nbins = _get_chrdatasize(chrname, genomeref, genomedatadir, res)
    # Create mapping arrays
    sliceselections, sliceweights, slicetopixels, pixelweights, mapping = \
                    _calc_subsetmappingdata(regionselection, baseres, res)
    # Initialize CG data array
    npx = len(mapping)
    fmat = np.zeros((npx, npx))
    # Map data to fmat
    minpos = np.min(sliceselections)
    maxpos = np.max(sliceselections)
    f = open(hicfile, 'r')
    for line in f:
        ## Increment pixel and the symmetric element
        i, j, fij = line.split()
        pos1 = int(i)
        if pos1 < minpos:
            continue
        pos2 = int(j)
        if pos2 < minpos:
            continue
        elif pos2 > maxpos and pos1 > maxpos:
            break
        val = float(fij)
        if pos1 in sliceselections and pos2 in sliceselections:
            x = sliceselections.index(pos1)
            y = sliceselections.index(pos2)
            fmat[slicetopixels[x], slicetopixels[y]] += \
                            val * sliceweights[x] * sliceweights[y]
            if pos1 != pos2:
                fmat[slicetopixels[y], slicetopixels[x]] += \
                                val * sliceweights[x] * sliceweights[y]
    f.close()
    # Normalization
    if norm == 'raw':
        pass
    elif norm == 'KR' or norm == 'VC' or norm == 'SQRTVC':
        # Read norm vector
        normfile = hicfile[:hicfile.find('RAWobserved')] + norm + 'norm'
        normvec = []
        with open(normfile, 'r') as f:
            for line in f:
                val = float(line)
                # If 0.0 or nan, set to inf
                if val == 0.0 or not np.isfinite(val):
                    val = np.inf
                normvec.append(val)
        normvec = np.array(normvec)
        # Truncate normvec
        normvec = normvec[:len(fmat)]
        # Divide fmat by outer product
        fmat /= np.outer(normvec, normvec)
    else:
        print 'Invalid normalization mode', norm, '!'
        sys.exit(1)
    hasdata = (pixelweights > 0.0) * (np.sum(fmat, 0) > 0.0)
    if np.sum(hasdata) < len(pixelweights):
        pixelweights = pixelweights[hasdata]
        mapping = np.array(mapping)[hasdata]
        fmat = fmat[hasdata][:, hasdata]
        print
    if weightboundaries:
        fmat /= np.outer(pixelweights, pixelweights)
    fmat += (nloop - 1.0) * np.diag(np.diag(fmat))
    # Return
    return fmat, (mapping, nbins)


## Reading binary archives


def _get_runbinarydir(pars):
    """
    Get directory pointing to binary data archive.
    """
    rundir = pars.get('rundir', 'rundata')
    chrref = pars.get('chrref', None)
    region = pars.get('region', 'full')
    accession = pars.get('accession', None)
    runlabel = pars.get('runlabel', None)
    beta = pars.get('beta', 1.0)
    resname = pars.get('resname', str(pars.get('res', None) / 1000) + 'kb')
    minrowsum = pars.get('minrowsum', 1.0)
    nloop = pars.get('nloop', 0)
    if chrref is None or accession is None or runlabel is None or \
        resname is None:
            print 'Insufficient input parameters to find binary directory!'
            return None
    labellist = [accession, runlabel]
    if np.abs(beta - 1.0) > 1.0e-6:
        labellist.append(('beta%.1lf' % beta))
    if minrowsum > 1.0:
        labellist.append(('rowsum%.1e' % int(minrowsum)))
    if nloop > 0:
        labellist.append(('nloop%i' % int(nloop)))
    binarydir = os.path.join(rundir, chrref, region,
                    '-'.join(labellist), resname)
    return binarydir


def _get_runbinarydir_interchr(pars):
    """
    Get directory pointing to binary data archive.
    For inter-chromosomal interactions.
    """
    rundir = pars['rundir']
    accession = pars['accession']
    runlabel = pars['runlabel']
    chrfullname1 = pars['chrfullname1']
    chrfullname2 = pars['chrfullname2']
    region = pars.get('region', 'full')
    res = pars['res']
    beta = pars.get('beta', 1.0)
    norm = pars.get('norm', 'raw')
    minrowsum = pars.get('minrowsum', 1.0)
    nloop = pars.get('nloop', 0)
    resname = str(res / 1000) + 'kb' if res < 1000000 else \
                    str(res / 1000000) + 'mb'
    labellist = [accession, runlabel]
    if np.abs(beta - 1.0) > 1.0e-6:
        labellist.append(('beta%.1lf' % beta))
    if minrowsum > 1.0:
        labellist.append(('rowsum%.1e' % int(minrowsum)))
    if nloop > 0:
        labellist.append(('nloop%i' % int(nloop)))
    binarydir = os.path.join(rundir, '_'.join([chrfullname1, chrfullname2]),
                    region, '-'.join(labellist), resname)
    if not os.path.exists(binarydir):
        os.makedirs(binarydir)
    return binarydir


def _get_mappingdata(datadir, norm='raw'):
    """
    Get mapping array from file.
    """
    fname = os.path.join(datadir, 'mapping-' + norm + '.dat')
    mapping, nbins = pickle.load(open(fname, 'rb'))
    return mapping, nbins


def _get_allchrmappedsizes(pars, chrfullnamelist, minrowsumdict=None):
    """
    Find out total number of mapped pixels in data,
    for all chromosomes in list.
    """
    parstemp = copy.deepcopy(pars)
    datavals = []
    for chrfullname in chrfullnamelist:
        parstemp['chrfullname'] = chrfullname
        parstemp['chrref'] = chrfullname
        parstemp['minrowsum'] = 0.5 if minrowsumdict is None \
                        else minrowsumdict[chrfullname]
        mapping = _get_mappingdata(_get_runbinarydir(parstemp),
                        norm=pars['norm'])[0]
        datavals.append(len(mapping))
    return np.array(datavals)


def _get_mmat(datadir, norm='raw'):
    """
    Calculate committor from MFPT.
    """
    # Check if cmat data exists
    mmatfile = os.path.join(datadir, 'mmat-' + norm + '.dat')
    if os.path.isfile(mmatfile):
        mmat = np.fromfile(mmatfile, 'float64')
        mmatlen = int(np.sqrt(len(mmat)))
        mmat.shape = mmatlen, mmatlen
    else:
        print 'Compute MFPT...'
        fmatfile = os.path.join(datadir, 'fmat-' + norm + '.dat')
        fmat = np.fromfile(fmatfile, 'float64')
        fmatlen = int(np.sqrt(len(fmat)))
        fmat.shape = fmatlen, fmatlen
        mmat = _calc_MFPT(fmat)
        fname = os.path.join(datadir, 'mmat-' + norm + '.dat')
        mmat.tofile(fname)
    return mmat


def _get_arrays(rundatadir, norm='raw'):
    fname = os.path.join(rundatadir, 'cmat-' + norm + '.dat')
    if not os.path.isfile(fname):
        cmat = None
    else:
        cmat = np.fromfile(fname, 'float64')
        nbins = int(np.sqrt(len(cmat)))
        cmat.shape = nbins, nbins
    fname = os.path.join(rundatadir, 'fmat-' + norm + '.dat')
    fmat = np.fromfile(fname, 'float64')
    nbins = int(np.sqrt(len(fmat)))
    fmat.shape = nbins, nbins
    fname = os.path.join(rundatadir, 'mmat-' + norm + '.dat')
    if not os.path.isfile(fname):
        mmat = None
    else:
        mmat = np.fromfile(fname, 'float64')
        nbins = int(np.sqrt(len(mmat)))
        mmat.shape = nbins, nbins
    mappingdata = _get_mappingdata(rundatadir, norm)
    return fmat, mmat, cmat, mappingdata

## Handling target set dict data


def _get_rhotsetdicts(plotdir, prefix):
    fname = plotdir + prefix + '-rhodict.p'
    fnameflag = fname + '-open'
    if os.path.isfile(fname):
        while True:
            if not os.path.isfile(fnameflag):
                open(fnameflag, 'a').close()
                break
            print 'Waiting for data dict lock to free up...'
            sleep(1)
    rhodict = pickle.load(open(fname, 'rb'))
    fname = plotdir + prefix + '-tsetdict.p'
    targetsetdict = pickle.load(open(fname, 'rb'))
    os.remove(fnameflag)
    return rhodict, targetsetdict


def _get_rhotsetdicts_20160801(tsetdatadir, tsetdataprefix,
                chrfullname, region, res, dataset, free=True, rhomodesfx=''):
    """
    Data dict reader
    Version 20160801: If free=False, file lock will not be freed after reading.
    """
    resname = str(res / 1000) + 'kb'
    dirname = os.path.join(tsetdatadir, tsetdataprefix,
                    chrfullname, region, resname)
    outrfname = os.path.join(dirname, dataset + '-rhodict' +
                            rhomodesfx + '.p')
    outtfname = os.path.join(dirname, dataset + '-tsetdict' +
                            rhomodesfx + '.p')
    if os.path.isdir(dirname) and os.path.isfile(outrfname) \
                    and os.path.isfile(outtfname):
        # Read files
        rdict, tdict = _pickle_securereads([outrfname, outtfname], free=free)
        #rdict = _pickle_secureread(outrfname, free=free)
        #tdict = _pickle_secureread(outtfname, free=free)
    else:
        # Create directory, new dicts
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        rdict = {}
        tdict = {}
        if not free:
            fname2 = outrfname + '-lock'
            open(fname2, 'a').close()
    return rdict, tdict


def _update_ConstructMC(rhodata, targetsetdata, plotdir, prefix):
    """
    Update database with output from run_ConstructMC_fullarray
    """
    fname = plotdir + prefix + '-rhodict.p'
    fnameflag = fname + '-open'
    if os.path.isfile(fname):
        while True:
            if not os.path.isfile(fnameflag):
                open(fnameflag, 'a').close()
                break
            else:
                print 'Waiting for lock on data dict to free up...'
                sleep(1)
        print 'Database update'
        rhodict0 = pickle.load(open(fname, 'rb'))
        fname = plotdir + prefix + '-tsetdict.p'
        targetsetdict0 = pickle.load(open(fname, 'rb'))
        # Check if new results are better
        for key in rhodata:
            if key in rhodict0:
                oldrho = rhodict0[key]
                newrho = rhodata[key]
                if oldrho < newrho:
                    rhodata[key] = rhodict0[key]
                    targetsetdata[key] = targetsetdict0[key]
                elif oldrho > newrho and not \
                        np.allclose(targetsetdict0[key], targetsetdata[key]):
                    print 'Improved', key
            else:
                print 'Create data', key
        rhodict0.update(rhodata)
        targetsetdict0.update(targetsetdata)
        fname = plotdir + prefix + '-rhodict.p'
        pickle.dump(rhodict0, open(fname, 'wb'))
        fname = plotdir + prefix + '-tsetdict.p'
        pickle.dump(targetsetdict0, open(fname, 'wb'))
        rhodata = rhodict0
        targetsetdata = targetsetdict0
    else:
        print 'Database creation'
        for key in rhodata:
            print 'Create data', key
        fname = plotdir + prefix + '-rhodict.p'
        pickle.dump(rhodata, open(fname, 'wb'))
        fname = plotdir + prefix + '-tsetdict.p'
        pickle.dump(targetsetdata, open(fname, 'wb'))
    os.remove(fnameflag)
    return rhodata, targetsetdata


def _update_ConstructMC_20160801(rdata, tsetdata, tsetdatadir, tsetdataprefix,
                chrfullname, region, res, dataset, rhomode='frac'):
    """
    Update database with output from run_ConstructMC_fullarray
    Version 20160801: Split data dicts
    """
    rhomodesfx = _get_rhomodesfx(rhomode)
    rdict, tdict = _get_rhotsetdicts_20160801(tsetdatadir, tsetdataprefix,
                chrfullname, region, res, dataset, free=False,
                rhomodesfx=rhomodesfx)
    # Check if new results are better
    for key in rdata:
        if key in rdict and key not in tdict:
            print 'Key error: Erase', key
            del(rdict[key])
        if (key not in rdict) or (rdict[key] > rdata[key]):
            if key not in rdict:
                print 'Create data', key
            else:
                print 'Improved', key
            # Update
            rdict[key] = rdata[key]
            tdict[key] = tsetdata[key]
    # Write to file
    resname = str(res / 1000) + 'kb'
    dirname = os.path.join(tsetdatadir, tsetdataprefix,
                    chrfullname, region, resname)
    rfname = os.path.join(dirname, dataset + '-rhodict' + rhomodesfx + '.p')
    tfname = os.path.join(dirname, dataset + '-tsetdict' + rhomodesfx + '.p')
    _pickle_securedumps((rfname, tfname), (rdict, tdict), freed=False)
    return


def _get_tsetdataset2(pars):
    """
    Get string identifying corresponding data set in tset dict.
    """
    accession = pars.get('accession', None)
    runlabel = pars.get('runlabel', None)
    threshold = pars.get('threshold', 0.0)
    norm = pars.get('norm', 'raw')
    minrowsum = pars.get('minrowsum', 1.0)
    nloop = pars.get('nloop', 0)
    if accession is None or runlabel is None:
        print 'Insufficient input parameters to find tset dataset!'
        return None
    datasetlist = ['-'.join([accession, runlabel])]
    if threshold >= 1.0:
        datasetlist.append('th%i' % int(threshold))
    if minrowsum > 1.0:
        datasetlist.append(('rowsum%.1e' % int(minrowsum)))
    if nloop > 0:
        datasetlist.append(('nloop%i' % int(nloop)))
    if norm != 'raw':
        datasetlist.append(norm)
    dataset2 = '-'.join(datasetlist)
    return dataset2


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


def _get_tset(pars, tsetdict):
    """
    Get selected target set from single chromosome.
    """
    region = pars.get('region', 'full')
    beta = pars.get('beta', 1.0)
    res = pars['res']
    chrfullname = pars['chrfullname']
    ntarget = pars['ntarget']
    tsetdataset = _get_tsetdataset2(pars)
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
    tsetdataset = _get_tsetdataset2(pars)
    key = (tsetdataset, chrfullname, region, beta, res, ntarget)
    if key not in rhodict:
        return np.nan
    rho = rhodict[key]
    return rho


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
    tsetdataset = _get_tsetdataset2(pars)
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


def _get_tsetReference_20180802(pars, ntarget):
    """
    Get reference target set for pars['chrfullname'].
    """
    tsetdatadir = pars['tsetdatadir']
    tsetdataprefix = pars['tsetdataprefix']
    region = pars.get('region', 'full')
    beta = pars.get('beta', 1.0)
    res = pars['res']
    resname = str(res / 1000) + 'kb'
    chrfullname = pars['chrfullname']
    tsetdataset = _get_tsetdataset2(pars)
    dirname = os.path.join(tsetdatadir, tsetdataprefix,
                    chrfullname, region, resname)
    fname = os.path.join(dirname, tsetdataset +
                    '-reftsetdict_20160802.p')
    tsetdict = _pickle_secureread(fname, free=True)
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
    tsetdataset = _get_tsetdataset2(pars)
    key1 = (tsetdataset, chrfullname1, region, beta, res, ntarget1)
    key2 = (tsetdataset, chrfullname2, region, beta, res, ntarget2)
    tset1 = tsetdict[key1]
    tset2 = tsetdict[key2]
    return tset1, tset2

## Accessing tset-related binary data


def _get_tsetmapkey(datadir, tset, norm='raw'):
    """
    Get unique identifier used to label target sets on the same Hi-C array.
    If tset is not currently mapped, give new ID.
    """
    # Get and lock tsetmap
    fname = os.path.join(datadir, 'tsetmap.p')
    tsetmap = _pickle_secureread(fname, free=False)
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
        _pickle_securedump(fname, tsetmap, freed=freed)
    else:
        # Remove lock
        fname2 = fname + '-lock'
        os.remove(fname2)
    return tsetindex


def _get_TargetCommittor(datadir, tset, norm='raw'):
    """
    Wrapper to obtain effective Laplacian between targets: Either retrieve
    from file, or compute from scratch.
    """
    # Get and lock tsetmap
    if not np.allclose(np.sort(tset), np.array(tset)):
        print 'Error: tset must be sorted!'
        sys.exit(1)
    #print tset
    fname = os.path.join(datadir, 'tsetmap.p')
    tsetmap = _pickle_secureread(fname, free=False)
    if tsetmap is None:
        freed = True
        tsetmap = {}
    else:
        freed = False
    tsetindex = None
    tsetorder = None
    for index, (targetset, n) in tsetmap.iteritems():
        if len(targetset) != len(tset):
            continue
        if np.allclose(targetset, tset) and n == norm:
            tsetindex = index
            tsetorder = np.argsort(targetset)
            break
    if tsetindex is None:
        #print 'Create new entry in tsetmap...'
        #print fname
        targetset = copy.deepcopy(tset)
        # Update dict
        tsetindex = len(tsetmap)
        tsetmap[tsetindex] = tset, norm
        _pickle_securedump(fname, tsetmap, freed=freed)
    else:
        # Remove lock
        fname2 = fname + '-lock'
        os.remove(fname2)
    # Is data file available?
    fname_TC = os.path.join(datadir,
            'TargetCommittor-%s-%02i.dat' % (norm, tsetindex))
    if os.path.isfile(fname_TC):
        # Read file
        print 'Load committor from file...'
        ntarget = len(tset)
        qAi = np.fromfile(fname_TC, 'float64')
        qAi.shape = ntarget, len(qAi) / ntarget
        if tsetorder is not None:
            qAi = qAi[tsetorder]
    else:
        print 'Compute committor...'
        # Compute tset and dump to file...
        qAi = _calc_qAi_sort_exact(datadir, targetset, norm=norm)
        qAi.tofile(fname_TC)
    return qAi


def _get_TargetEffLaplacian(datadir, tset, norm='raw'):
    """
    Wrapper to obtain effective Laplacian between targets: Either retrieve
    from file, or compute from scratch.
    """
    if np.sort(tset) != np.array(tset):
        print 'Error: tset must be sorted!'
        sys.exit(1)
    # Get and lock tsetmap
    fname = os.path.join(datadir, 'tsetmap.p')
    tsetmap = _pickle_secureread(fname, free=False)
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
        _pickle_securedump(fname, tsetmap, freed=freed)
    else:
        # Remove lock
        fname2 = fname + '-lock'
        os.remove(fname2)
    # Is data file available?
    fname_TEL = os.path.join(datadir,
            'TargetEffLaplacian-%s-%02i_20160628.dat' % (norm, tsetindex))
    if os.path.isfile(fname_TEL):
        # Read file
        print 'Load effective Laplacian from file...'
        ntarget = len(tset)
        lab = np.fromfile(fname_TEL, 'float64').reshape(ntarget, ntarget)
    else:
        # Compute tset and dump to file...
        lab = _calc_TargetEffLaplacian(datadir, tset, norm=norm)
        lab.tofile(fname_TEL)
    return lab


def _get_TargetEffLaplacian_20160802(datadir, tset, norm='raw'):
    """
    Wrapper to obtain effective Laplacian between targets: Either retrieve
    from file, or compute from scratch.
    """
    # Get and lock tsetmap
    fname = os.path.join(datadir, 'tsetmap.p')
    tsetmap = _pickle_secureread(fname, free=False)
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
        _pickle_securedump(fname, tsetmap, freed=freed)
    else:
        # Remove lock
        fname2 = fname + '-lock'
        os.remove(fname2)
    # Is data file available?
    fname_TEL = os.path.join(datadir,
            'TargetEffLaplacian-%s-%02i_20160802.dat' % (norm, tsetindex))
    if os.path.isfile(fname_TEL):
        # Read file
        print 'Load effective Laplacian from file...'
        ntarget = len(tset)
        lab = np.fromfile(fname_TEL, 'float64').reshape(ntarget, ntarget)
    else:
        # Compute tset and dump to file...
        lab = _calc_TargetEffLaplacian_20160802(datadir, tset, norm=norm)
        lab.tofile(fname_TEL)
    return lab


def _get_TargetEffLaplacian_20160802_mixbeta(pars, tset, norm='raw'):
    """
    Wrapper to obtain effective Laplacian between targets: Either retrieve
    from file, or compute from scratch.
    Note: Use different beta for tset (qAi computation) and fmat.
          - Define pars['tsetbeta'], pars['fmatbeta'].
    """
    # Get and lock tsetmap
    pars['beta'] = pars['tsetbeta']
    tsetdatadir = _get_runbinarydir(pars)
    tsetmapping, _ = _get_mappingdata(_get_runbinarydir(pars), norm=norm)
    pars['beta'] = pars['fmatbeta']
    fmatdatadir = _get_runbinarydir(pars)
    fmatmapping, _ = _get_mappingdata(_get_runbinarydir(pars), norm=norm)
    mappedtset = tsetmapping[tset]
    fname = os.path.join(tsetdatadir, 'tsetmap.p')
    tsetmap = _pickle_secureread(fname, free=False)
    if tsetmap is None:
        freed = True
        tsetmap = {}
    else:
        freed = False
    tsetindex = None
    for index, (targetset, n) in tsetmap.iteritems():
        if len(targetset) != len(tset):
            continue
        if np.allclose(targetset, tset) and n == norm:
            tsetindex = index
            break
    if tsetindex is None:
        print 'Compute effective Laplacian...'
        # Update dict
        tsetindex = len(tsetmap)
        tsetmap[tsetindex] = tset, norm
        _pickle_securedump(fname, tsetmap, freed=freed)
    else:
        # Remove lock
        fname2 = fname + '-lock'
        os.remove(fname2)
    # Map tset back to fmatbeta mappingdata
    if not set(mappedtset).issubset(set(fmatmapping)) or not \
            set(tsetmapping).issubset(set(fmatmapping)):
        print 'Cannot unmap tset!! tsetbeta=%.1lf, fmatbeta=%.1lf, ntarget=%i' \
                    % (pars['tsetbeta'], pars['fmatbeta'], pars['ntarget'])
        return None
    tset2 = [list(fmatmapping).index(t) for t in mappedtset]
    tset2fmatmapping = np.array([list(fmatmapping).index(t)
                    for t in tsetmapping])
    # Is data file available?
    print 'tsetindex: %i' % tsetindex
    print 'tset:', tset
    print 'tset2:', tset2
    fname_TEL = os.path.join(tsetdatadir,
            'TargetEffLaplacian-%s-%02i_fmatbeta%.1lf_20160802.dat' %
            (norm, tsetindex, pars['fmatbeta']))
    if os.path.isfile(fname_TEL):
        # Read file
        print 'Load effective Laplacian from file...'
        ntarget = len(tset2)
        lab = np.fromfile(fname_TEL, 'float64').reshape(ntarget, ntarget)
    else:
        # Compute tset and dump to file...
        qAi = _get_TargetCommittor(tsetdatadir, tset, norm=norm)
        qAip = np.zeros((len(qAi), len(fmatmapping)))
        qAip[:, tset2fmatmapping] = qAi
        fmat, _, _, _ = _get_arrays(fmatdatadir, norm=norm)
        fmat -= np.diag(np.diag(fmat))
        fmat /= np.sum(fmat)
        tset2 = np.sort(tset2)
        ntarget = len(tset2)
        pivec = np.sum(fmat, axis=0) / np.sum(fmat)
        pmat = np.array([v / pivec[i] for i, v in enumerate(fmat)])
        lmat = pmat - np.diag(np.ones_like(pivec))
        lmat2 = lmat[:, tset2]
        qAi2 = qAip * np.array([list(pivec)] * ntarget)
        pivec2 = np.dot(qAip, pivec)
        lab = np.dot(qAi2, lmat2)
        lab.tofile(fname_TEL)
    return lab


def _get_TargetEffLaplacian_interchr_20160802(pars):
    """
    Wrapper to obtain effective Laplacian between targets: Either retrieve
    from file, or compute from scratch.
    Inter-chromosomal case, with 'unmapped' tsets.
    """
    # Get tsets
    tpars = copy.deepcopy(pars)
    chrfullname1 = pars['chrfullname1']
    chrfullname2 = pars['chrfullname2']
    ntarget1 = pars['ntarget1']
    ntarget2 = pars['ntarget2']
    norm = pars['norm']
    tpars['chrfullname'] = chrfullname1
    mappedtset1 = _get_tsetReference_20180802(tpars, ntarget1)[0]
    tpars['chrref'] = chrfullname1
    mapping, _ = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    tset1 = np.array([list(mapping).index(t) for t in mappedtset1])
    tpars['chrfullname'] = chrfullname2
    mappedtset2 = _get_tsetReference_20180802(tpars, ntarget2)[0]
    tpars['chrref'] = chrfullname2
    mapping, _ = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    tset2 = np.array([list(mapping).index(t) for t in mappedtset2])
    # Get data directory
    datadir = _get_runbinarydir_interchr(pars)
    # Get and lock tsetmap
    fname = os.path.join(datadir, 'tsetmap.p')
    tsetmap = _pickle_secureread(fname, free=False)
    if tsetmap is None:
        freed = True
        tsetmap = {}
    else:
        freed = False
    tsetindex = None
    for index, (targetset1, targetset2, n) in tsetmap.iteritems():
        if (len(targetset1) != len(tset1)) or (len(targetset2) != len(tset2)):
            continue
        if set(targetset1) == set(tset1) and \
                set(targetset2) == set(tset2) and n == norm:
            tsetindex = index
            break
    if tsetindex is None:
        print 'Compute effective Laplacian...'
        # Update dict
        tsetindex = len(tsetmap)
        tsetmap[tsetindex] = tset1, tset2, norm
        _pickle_securedump(fname, tsetmap, freed=freed)
    else:
        # Remove lock
        print 'tsetmap index found:', tsetindex
        fname2 = fname + '-lock'
        os.remove(fname2)
    # Is data file available?
    fname_TEL = os.path.join(datadir,
            'TargetEffLaplacian-%s-%02i.dat' % (norm, tsetindex))
    if os.path.isfile(fname_TEL):
        # Read file
        print 'Load effective Laplacian from file...'
        ntarget1, ntarget2 = len(tset1), len(tset2)
        lab = np.fromfile(fname_TEL, 'float64').reshape(ntarget1, ntarget2)
    else:
        # Compute tset and dump to file...
        lab = _calc_TargetEffLaplacian_interchr_20160802(pars)
        lab.tofile(fname_TEL)
    return lab

## Access inter-chromosomal binary data


def _get_fmatmap_inter(pars):
    """
    Get inter-chromosomal interaction fmat, from binary file, or from
    sparse ASCII file.
    Also gives mapping data.
    Full chromosome selections only.
    """
    # Parse parameters
    #rundir = pars['rundir']
    #accession = pars['accession']
    #runlabel = pars['runlabel']
    #chrfullname1 = pars['chrfullname1']
    #chrfullname2 = pars['chrfullname2']
    #region = pars.get('region', 'full')
    #res = pars['res']
    #beta = pars.get('beta', 1.0)
    norm = pars.get('norm', 'raw')
    #dataset = '-'.join([accession, runlabel])
    #resname = str(res / 1000) + 'kb' if res < 1000000 else \
                    #str(res / 1000000) + 'mb'
    binarydir = _get_runbinarydir_interchr(pars)
    binaryfname = os.path.join(binarydir, 'fmat-' + norm + '.dat')
    mappingdatafname = os.path.join(binarydir,
                    'mapping-' + norm + '.dat')
    if os.path.isfile(binaryfname):
        # Read fmat, mappingdata
        fmat = np.fromfile(binaryfname, 'float64')
        mappingdatafname = os.path.join(binarydir,
                        'mapping-' + norm + '.dat')
        md1, md2 = pickle.load(open(mappingdatafname, 'rb'))
        fmat.shape = len(md1[0]), len(md2[0])
    else:
        print 'Inter-chromosome data not available!'
        print binarydir
        print 'Run 20_InterChr-Extractor.py ...'
        sys.exit()
    return fmat, md1, md2


##########################################################################
# Epigenetic data track extraction


def _get_epigendatatrack(epigenpars, getalpha=False):
    """
    Extract epigenetic data track from bigbed coverage / bigwig signal files.
    If getalpha is True, estimate signal-noise parameter alpha using Poisson
    model: readout = (1-alpha)*signal + alpha*noise
    """
    # Extract pars
    epigendatadir = epigenpars['epigendatadir']
    epigentypes = epigenpars['epigentypes']
    tracktypes = epigenpars['tracktypes']
    eptype = epigenpars['eptype']
    trtype = epigenpars['trtype']
    sig = epigenpars['signal']
    src = epigenpars['source']
    chrfullname = epigenpars['chrfullname']
    stpos = epigenpars['startpos']
    enpos = epigenpars['endpos']
    binsize = epigenpars['binsize']
    nbins = int((enpos - stpos) / binsize)
    # Get data file paths
    ## Read data file list
    if eptype in epigentypes and trtype in tracktypes:
        filetype = '-bigbeds' if trtype == 'coverage' else '-bigwigs'
        flistfname = os.path.join(epigendatadir, eptype + filetype + '.txt')
        if not os.path.isfile(flistfname):
            print 'File list %s not found!' % flistfname
            return None
        signals = []
        sources = []
        fnames = []
        with open(flistfname, 'r') as f:
            for line in f:
                vals = line.strip().split('\t')
                if len(vals) < 3:
                    break
                if vals[0][0] == '#':
                    continue
                signals.append(vals[0])
                sources.append(vals[1])
                fnames.append(vals[2])
    else:
        print 'Invalid epigentype %s, tracktype %s!' % (eptype, trtype)
        return None
    #print 'sources:', sources
    if trtype == 'coverage':
        cmdname = 'bigBedSummary'
    else:
        cmdname = 'bigWigSummary'
    if sig != 'all':
        ## Select file
        fnamechoice = None
        fnamebackup = None
        ### If source is not found, use the first track with the same data set
        for signal, source, fname in zip(signals, sources, fnames):
            if sig == signal:
                if src == '' or src == source:
                    src = source
                    fnamechoice = fname
                    break
                elif fnamebackup is None:
                    fnamebackup = fname
        if fnamechoice is None:
            print 'Using fnamebackup'
            fnamechoice = fnamebackup
        # Compose extraction command string
        fname = os.path.join(epigendatadir, fnamechoice)
        cmdstr = [cmdname, fname, chrfullname, str(stpos),
                        str(enpos), str(nbins)]
        # Run command string
        #print 'Running:', fname
        ## Execute subprocess to obtain data string
        datalist = subprocess.check_output(cmdstr).split()
        # Parse output data
        for a, v in enumerate(datalist):
            if v == 'n/a':
                datalist[a] = 0.0
            else:
                datalist[a] = float(v)
        trackdata = np.array(datalist)
        tracklabel = '%s %s (%s)' % (sig, trtype, src)
        if getalpha:
            # Find control data file
            if eptype == 'TFBS' and src == 'SYDH':
                cutindex = fnamechoice.index('Sig.bigWig')
                sigfnamecut = fnamechoice[:cutindex]
                if sigfnamecut.endswith('Std'):
                    sig2 = 'ControlStd'
                elif sigfnamecut.endswith('Iggmus'):
                    sig2 = 'ControlIggmus'
                elif sigfnamecut.endswith('Iggrab'):
                    sig2 = 'ControlIggrab'
                else:
                    print 'Can\'t find correct control data!'
                    return trackdata, tracklabel, np.nan
            elif eptype == 'Histone':
                sig2 = 'Control'
            else:
                print 'No control data available!'
                return trackdata, tracklabel, np.nan
            ## Select file
            fnamechoice = None
            fnamebackup = None
            ### If source is not found, use the first track with the same data set
            for signal, source, fname in zip(signals, sources, fnames):
                if sig2 == signal:
                    if src == '' or src == source:
                        src = source
                        fnamechoice = fname
                        break
                    elif fnamebackup is None:
                        fnamebackup = fname
            controlfname = copy.deepcopy(fnamechoice)
            # Compose extraction command string
            fname = os.path.join(epigendatadir, controlfname)
            cmdstr = [cmdname, fname, chrfullname, str(stpos),
                            str(enpos), str(nbins)]
            # Run command string
            print 'Running:', fname
            ## Execute subprocess to obtain data string
            datalist = subprocess.check_output(cmdstr).split()
            # Parse output data
            for a, v in enumerate(datalist):
                if v == 'n/a':
                    datalist[a] = 0.0
                else:
                    datalist[a] = float(v)
            ctrvec = np.array(datalist)
            tracklabel = '%s %s (%s)' % (sig, trtype, src)
            # Initialize parameters
            sigtot = np.sum(trackdata)
            ctrtot = np.sum(ctrvec)
            alphavals = [1.0]
            maxiter = 100
            # Iterate
            for i in range(1, maxiter):
                mask = (trackdata < (ctrvec * alphavals[-1] * sigtot / ctrtot))
                alphavals.append(np.sum(trackdata[mask]) / np.sum(ctrvec[mask]) *
                                ctrtot / sigtot)
                #print 'alpha=%.3e' % alphavals[-1]
                if alphavals[-1] > alphavals[-2] or np.abs(alphavals[-1] -
                                alphavals[-2]) < 1.0e-6 or \
                                alphavals[-1] == 0.0 or \
                                np.isnan(alphavals[-1]):
                    _ = alphavals.pop()
                    alpha = alphavals[-1]
                    break
            return trackdata, tracklabel, alpha
        return trackdata, tracklabel
    else:
        ## Extract all data tracks of the given eptype, trtype
        trackdatalist = []
        tracklabellist = []
        for signal, source, fname in zip(signals, sources, fnames):
            # Compose extraction command string
            fname2 = os.path.join(epigendatadir, fname)
            cmdstr = [cmdname, fname2, chrfullname, str(stpos),
                            str(enpos), str(nbins)]
            # Run command string
            #print 'Running:', ' '.join(cmdstr)
            ## Execute subprocess to obtain data string
            try:
                datalist = subprocess.check_output(cmdstr).split()
                for a, v in enumerate(datalist):
                    if v == 'n/a':
                        datalist[a] = 0.0
                    else:
                        datalist[a] = float(v)
                trackdatalist.append(np.array(datalist))
            except subprocess.CalledProcessError, e:
                print 'No data for %s %s (%s)' % (signal, trtype, source)
                trackdatalist.append(None)
            # Parse output data
            tracklabellist.append('%s %s (%s)' % (signal, trtype, source))
        return trackdatalist, tracklabellist


def PlotEpigeneticTrack_1(epigenpars, x):
    """
    Extract epigenetic data and plot on axis.
    """
    data = _get_epigendatatrack(epigenpars)
    if data is None:
        print 'Epigenetic track data extraction failed.'
        return None
    trackdata, tracklabel = data
    xvals = (np.arange(epigenpars['startpos'], epigenpars['endpos'],
                    epigenpars['binsize']) +
                    0.5 * epigenpars['binsize']) / 1.0e6
    x.plot(xvals, trackdata)
    x.set_title(tracklabel)
    x.set_xlim(epigenpars['startpos'] / 1.0e6, epigenpars['endpos'] / 1.0e6)
    return trackdata, tracklabel


def PlotEpigeneticTrack_all(epigenpars, x):
    """
    Extract epigenetic data and plot on axis.
    Extracts all tracks of specified type.
    """
    data = _get_epigendatatrack(epigenpars)
    if data is None:
        print 'Epigenetic track data extraction failed.'
        return None
    trackdatal, tracklabell = data
    # Clean out empty tracks
    trackdatal = np.array(trackdatal)
    tracklabell = np.array(tracklabell)
    goodinds = []
    for i, d in enumerate(trackdatal):
        if d is not None:
            goodinds.append(i)
    goodinds = np.array(goodinds)
    trackdatal = list(trackdatal[goodinds])
    tracklabell = list(tracklabell[goodinds])
    xvals = (np.arange(epigenpars['startpos'], epigenpars['endpos'],
                    epigenpars['binsize']) +
                    0.5 * epigenpars['binsize']) / 1.0e6
    for ax, data, label in zip(x, trackdatal, tracklabell):
        ax.plot(xvals, data)
        ax.set_title(label)
        ax.set_xlim(epigenpars['startpos'] / 1.0e6, epigenpars['endpos'] /
                        1.0e6)
    return trackdatal, tracklabell


def EpigeneticTracksReport_1Band(epigenbasepars, plots=True):
    """
    Create report on epigenetic track data, for single band.
    """
    # Parse parameters
    epigenpars = copy.deepcopy(epigenbasepars)
    chrfullname = epigenpars['chrfullname']
    bandname = epigenpars['bandname']
    ntrackpage = epigenpars['ntrackpage']
    eptypelist = epigenpars['eptypelist']
    trtypelist = epigenpars['trtypelist']
    clrslist = epigenpars['clrslist']
    banddatadir = epigenpars['banddatadir']
    tracks = []
    labels = []
    clrs = []
    # Get track data
    for eptype, trtype, clr in zip(eptypelist, trtypelist, clrslist):
        sig = 'all'
        src = ''
        epigenpars['eptype'] = eptype
        epigenpars['trtype'] = trtype
        epigenpars['signal'] = sig
        epigenpars['source'] = src
        trackdata, tracklabel = _get_epigendatatrack(epigenpars)
        trackdata = np.array(trackdata)
        tracklabel = np.array(tracklabel)
        goodinds = []
        for i, d in enumerate(trackdata):
            if d is not None:
                goodinds.append(i)
        goodinds = np.array(goodinds)
        trackdata = list(trackdata[goodinds])
        tracklabel = list(tracklabel[goodinds])
        tracks.extend(trackdata)
        labels.extend(tracklabel)
        clrs.extend([clr] * len(trackdata))
    # Plots
    if plots:
        xvals = (np.arange(epigenpars['startpos'], epigenpars['endpos'],
                    epigenpars['binsize']) +
                    0.5 * epigenpars['binsize']) / 1.0e6
        xlims = epigenpars['startpos'] / 1.0e6, epigenpars['endpos'] / 1.0e6
        ntrack = len(tracks)
        npage = int(np.ceil(ntrack / float(ntrackpage)))
        fname = os.path.join(banddatadir, '%s-%s.pdf' % (chrfullname, bandname))
        report = matplotlib.backends.backend_pdf.PdfPages(fname)
        for i in range(npage):
            f, x = plt.subplots(ntrackpage, 1, figsize=(10, 12))
            if (i + 1) * ntrackpage <= ntrack:
                ntrackthis = ntrackpage
                for j in range(ntrackpage):
                    y = tracks[i * ntrackpage + j]
                    npts = min(len(xvals), len(y))
                    x[j].plot(xvals[:npts], y[:npts],
                                    c=clrs[i * ntrackpage + j])
                    x[j].set_title(labels[i * ntrackpage + j])
                    x[j].set_xlim(xlims)
                    if j < ntrackpage - 1:
                        x[j].set_xticklabels(())
            else:
                ntrackthis = ntrack - i * ntrackpage
                for j in range(ntrackthis, ntrackpage):
                    f.delaxes(x[j])
                for j in range(ntrackthis):
                    y = tracks[i * ntrackpage + j]
                    npts = min(len(xvals), len(y))
                    x[j].plot(xvals[:npts], y[:npts],
                                    c=clrs[i * ntrackpage + j])
                    x[j].set_title(labels[i * ntrackpage + j])
                    x[j].set_xlim(xlims)
                    if j < ntrackthis - 1:
                        x[j].set_xticklabels(())
            suptitle = f.suptitle('Chromosome %s: %s' % (chrfullname[3:], bandname),
                            fontsize=18, y=1.02)
            plt.tight_layout()
            f.canvas.draw()
            report.savefig(f, bbox_extra_artists=(suptitle,), bbox_inches="tight")
            plt.close(f)
        report.close()
    return tracks, labels, clrs


def EpigeneticTracksReport_2Bands_1Chr(epigenbasepars):
    """
    Compare epigenetic track profiles of two bands on the same chromosome.
    """
    # Parse parameters
    epigenpars = copy.deepcopy(epigenbasepars)
    banddatadir = epigenpars['banddatadir']
    chrfullname = epigenpars['chrfullname']
    bandname1 = epigenpars['bandname1']
    bandname2 = epigenpars['bandname2']
    ntrackpage = epigenpars['ntrackpage']
    eptypelist = epigenpars['eptypelist']
    trtypelist = epigenpars['trtypelist']
    clrslist = epigenpars['clrslist']
    stpos1 = epigenpars['startpos1']
    enpos1 = epigenpars['endpos1']
    stpos2 = epigenpars['startpos2']
    enpos2 = epigenpars['endpos2']
    tracks1 = []
    labels = []
    clrs = []
    tracks2 = []
    #######################################
    # V2: Merge data sets, discarding only tracks with both empty
    # Get band1 / band2 data
    for eptype, trtype, clr in zip(eptypelist, trtypelist, clrslist):
        sig = 'all'
        src = ''
        epigenpars['startpos'] = stpos1
        epigenpars['endpos'] = enpos1
        epigenpars['eptype'] = eptype
        epigenpars['trtype'] = trtype
        epigenpars['signal'] = sig
        epigenpars['source'] = src
        trackdata1, tracklabel1 = _get_epigendatatrack(epigenpars)
        epigenpars['startpos'] = stpos2
        epigenpars['endpos'] = enpos2
        trackdata2, tracklabel2 = _get_epigendatatrack(epigenpars)
        # Find nonzero tracks
        for d1, l1, d2, l2 in zip(trackdata1, tracklabel1,
                        trackdata2, tracklabel2):
            if d1 is not None and d2 is not None:
                tracks1.append(d1)
                tracks2.append(d2)
                clrs.append(clr)
                labels.append(l1)
            elif d1 is None and d2 is not None:
                tracks2.append(d2)
                tracks1.append(np.zeros_like(np.arange(stpos1, enpos1,
                    epigenpars['binsize'])))
                clrs.append(clr)
                labels.append(l1)
            elif d2 is None and d1 is not None:
                tracks1.append(d1)
                tracks2.append(np.zeros_like(np.arange(stpos2, enpos2,
                    epigenpars['binsize'])))
                clrs.append(clr)
                labels.append(l1)
    #######################################
    # V1: Simply collect all tracks
    # Get band1 data
    #for eptype, trtype, clr in zip(eptypelist, trtypelist, clrslist):
        #sig = 'all'
        #src = ''
        #epigenpars['startpos'] = stpos1
        #epigenpars['endpos'] = enpos1
        #epigenpars['eptype'] = eptype
        #epigenpars['trtype'] = trtype
        #epigenpars['signal'] = sig
        #epigenpars['source'] = src
        #trackdata, tracklabel = _get_epigendatatrack(epigenpars)
        #tracks1.extend(trackdata)
        #labels.extend(tracklabel)
        #clrs.extend([clr] * len(trackdata))
    ## Get band2 data
    #for eptype, trtype, clr in zip(eptypelist, trtypelist, clrslist):
        #sig = 'all'
        #src = ''
        #epigenpars['startpos'] = stpos2
        #epigenpars['endpos'] = enpos2
        #epigenpars['eptype'] = eptype
        #epigenpars['trtype'] = trtype
        #epigenpars['signal'] = sig
        #epigenpars['source'] = src
        #trackdata, tracklabel = _get_epigendatatrack(epigenpars)
        #tracks2.extend(trackdata)
    xvals1 = (np.arange(stpos1, enpos1,
                    epigenpars['binsize']) +
                    0.5 * epigenpars['binsize']) / 1.0e6
    xvals1 = xvals1[:len(tracks1[0])]
    xlims1 = stpos1 / 1.0e6, enpos1 / 1.0e6
    xvals2 = (np.arange(stpos2, enpos2,
                    epigenpars['binsize']) +
                    0.5 * epigenpars['binsize']) / 1.0e6
    xvals2 = xvals2[:len(tracks2[0])]
    xlims2 = stpos2 / 1.0e6, enpos2 / 1.0e6
    ntrack = len(tracks1)
    npage = int(np.ceil(ntrack / float(ntrackpage)))
    fname = os.path.join(banddatadir, '%s-%s-%s-epigen.pdf' %
                    (chrfullname, bandname1, bandname2))
    report = matplotlib.backends.backend_pdf.PdfPages(fname)
    for i in range(npage):
        widthratio = float(enpos1 - stpos1) / (enpos1 - stpos1 +
                        enpos2 - stpos2)
        widthratio = [widthratio, 1.0 - widthratio]
        f = plt.figure(figsize=(11, 12))
        gs = gridspec.GridSpec(ntrackpage, 2, width_ratios=widthratio)
        if (i + 1) * ntrackpage <= ntrack:
            ntrackthis = ntrackpage
            x1 = [plt.subplot(gs[i2, 0]) for i2 in range(ntrackthis)]
            x2 = [plt.subplot(gs[i2, 1]) for i2 in range(ntrackthis)]
            for j in range(ntrackpage):
                maxv = max(np.max(tracks1[i * ntrackpage + j]),
                                np.max(tracks2[i * ntrackpage + j]))
                y = tracks1[i * ntrackpage + j]
                npts = min(len(xvals1), len(y))
                x1[j].plot(xvals1[:npts], y[:npts],
                                c=clrs[i * ntrackpage + j])
                x1[j].set_title(labels[i * ntrackpage + j], loc='left')
                x1[j].set_xlim(xlims1)
                x1[j].set_ylim(0, maxv)
                y = tracks2[i * ntrackpage + j]
                npts = min(len(xvals2), len(y))
                x2[j].plot(xvals2[:npts], y[:npts],
                                c=clrs[i * ntrackpage + j])
                x2[j].set_xlim(xlims2)
                x2[j].set_ylim(0, maxv)
                x2[j].set_yticklabels(())
                if j < ntrackpage - 1:
                    x1[j].set_xticklabels(())
                    x2[j].set_xticklabels(())
        else:
            ntrackthis = ntrack - i * ntrackpage
            x1 = [plt.subplot(gs[i2, 0]) for i2 in range(ntrackthis)]
            x2 = [plt.subplot(gs[i2, 1]) for i2 in range(ntrackthis)]
            for j in range(ntrackthis):
                maxv = max(np.max(tracks1[i * ntrackpage + j]),
                                np.max(tracks2[i * ntrackpage + j]))
                x1[j].plot(xvals1, tracks1[i * ntrackpage + j],
                                c=clrs[i * ntrackpage + j])
                x1[j].set_title(labels[i * ntrackpage + j], loc='left')
                x1[j].set_xlim(xlims1)
                x1[j].set_ylim(0, maxv)
                x2[j].plot(xvals2, tracks2[i * ntrackpage + j],
                                c=clrs[i * ntrackpage + j])
                x2[j].set_xlim(xlims2)
                x2[j].set_ylim(0, maxv)
                x2[j].set_yticklabels(())
                if j < ntrackthis - 1:
                    x1[j].set_xticklabels(())
                    x2[j].set_xticklabels(())
        suptitle = f.suptitle('Chromosome %s: %s vs. %s' %
                        (chrfullname[3:], bandname1, bandname2),
                        fontsize=16, y=1.02)
        plt.tight_layout()
        f.canvas.draw()
        report.savefig(f, bbox_extra_artists=(suptitle,), bbox_inches="tight")
        plt.close(f)
    report.close()
    return (tracks1, tracks2), labels, clrs


def _get_cancerMutVector_somaticCA_chr(pars):
    """
    Get cancer mutation density data, on single chromosome.
    """
    chrfullname = pars['chrfullname']
    norm = pars.get('norm', 'raw')
    cancertrack = pars['cancertrack']
    fname = os.path.join('epigenomic-tracks', 'CancerMutations',
                cancertrack + '-c' + chrfullname[3:] + '-flat.dat')
    arr = np.fromfile(fname, 'float64')
    _, nbins = _get_mappingdata(_get_runbinarydir(pars),
                                    norm=norm)
    if len(arr) < nbins:
        arr2 = np.zeros(nbins)
        arr2[:len(arr)] = arr
        return arr2
    elif len(arr) > nbins:
        arr2 = arr[:nbins]
        return arr2
    else:
        return arr


# Epigenetic data analysis


def _correlatevectors_2dweights(xvec, yvec, wmat):
    """
    Calculate weighted correlation between two vectors xvec and yvec, with
    pairs sampled with weight matrix wmat.
    """
    nx, ny = wmat.shape
    if nx != len(xvec) or ny != len(yvec):
        print 'Invalud dimensions: %i, %i, (%i, %i)' % \
                        (len(xvec), len(yvec), nx, ny)
    wmat2 = wmat / np.sum(wmat)
    mx = np.sum(np.array([list(xvec)] * ny).T * wmat2)
    my = np.sum(np.array([list(yvec)] * nx) * wmat2)
    cv12 = np.sum(wmat2 * np.outer(xvec - mx, yvec - my))
    cv11 = np.sum(wmat2 * (np.array([list(xvec - mx)] * ny) ** 2).T)
    cv22 = np.sum(wmat2 * np.array([list(yvec - my)] * nx) ** 2)
    return cv12 / np.sqrt(cv11 * cv22)


# GC content data extractor


def _get_GCvector_region(pars, sampledist=None, windowsize=2.0, region=None,
            ignoreN=True):
    """
    Gets average GC content (percentage) in specified region along chromosome.
    Notes:
        - Uses pars['chrfullname'] to specify chromosome.
        - sampledist: Distance between sampling points, in bp.
                      Default: pars['res']
        - windowsize: Size of averaging window, in multiples of sampledist
                      Default: 2.0
        - region: Genomic region to plot, in bp.
                  Default: full chromosome.
        - ignoreN: Remove 'N' characters from sequence string in GC computation.
    """
    genomedatadir = pars['genomedatadir']
    genomeref = pars['genomeref']
    rundir = pars['rundir']
    chrfullname = pars['chrfullname']
    if sampledist is None:
        sampledist = pars['res']
    windowsize2 = windowsize * sampledist
    # Check if binary vector exists
    fname = os.path.join(rundir, chrfullname, 'gc-%.e-%.1lf.dat' %
                         (sampledist, windowsize))
    if os.path.isfile(fname):
        gcarray = np.fromfile(fname, 'float64')
    else:
        # Get sequence record
        fname = os.path.join(genomedatadir, genomeref, '%s.fa' % chrfullname)
        handle = open(fname, 'r')
        gen = SeqIO.parse(handle, 'fasta')
        for rec in gen:
            pass
        chlen = len(rec)
        if region is None:
            region = [0, chlen]
        nbins = (region[1] - region[0]) / sampledist + 1
        gcarray = np.zeros(nbins)
        for i in range(nbins):
            lo = max(region[0], i * sampledist - int(windowsize2 / 2))
            hi = min(region[1], i * sampledist + int(windowsize2 / 2))
            if ignoreN:
                gccount = GC(rec[lo:hi].seq) * (hi - lo)
                nN = np.sum(np.array(list(rec[lo:hi].seq)) == 'N')
                gcarray[i] = gccount / (hi - lo - nN) if nN < (hi - lo) else 0.0
            else:
                gcarray[i] = GC(rec[lo:hi].seq)
        fname = os.path.join(rundir, chrfullname, 'gc-%.e-%.1lf.dat' %
                             (sampledist, windowsize))
        gcarray.tofile(fname)
    return gcarray


def _get_GCvalue_region(pars, region=None, ignoreN=True):
    """
    Gets average GC content (percentage) in specified region along chromosome.
    Notes:
        - Uses pars['chrfullname'] to specify chromosome.
        - region: Genomic region to sample, in bp.
                  Default: full chromosome.
        - ignoreN: Remove 'N' characters from sequence string in GC computation.
    """
    genomedatadir = pars['genomedatadir']
    genomeref = pars['genomeref']
    chrfullname = pars['chrfullname']
    # Get sequence record
    fname = os.path.join(genomedatadir, genomeref, '%s.fa' % chrfullname)
    handle = open(fname, 'r')
    gen = SeqIO.parse(handle, 'fasta')
    for rec in gen:
        pass
    chlen = len(rec)
    if region is None:
        region = [0, chlen]
    lo, hi = region
    if ignoreN:
        gccount = GC(rec[lo:hi].seq) * (hi - lo)
        #nN = np.sum(np.array(list(rec[lo:hi].seq)) == 'N')
        nN = rec[lo:hi].seq.count('N')
        return gccount / (hi - lo - nN) if nN < (hi - lo) else 0.0
    else:
        return GC(rec[lo:hi].seq)


##########################################################################
# General computation / analysis functions
## Basic graph / MSM computations


def _LEMAnalysis(data):
    """
    Basic Laplacian eigenmap analysis: Identify structural hierarchies.
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


def _calc_MFPT(fmat):
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
    for j in range(nbins):
        ## Temp pmat
        pmatt = pmat.copy()
        pmatt[:, j] = 0.0
        mmat[:, j] = solve(pmatt - np.eye(nbins), -np.ones(nbins))
    return mmat


def _calc_MFPT_withLoops(fmat):
    """
    Calculate Markov mean first pass time.
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
        fmat = fmat[goodinds][:, goodinds]
        mmat = mmat[goodinds][:, goodinds]
        statprob = np.sum(fmat, 0)
    return fmat, mmat, mapping


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


## Metastability / target computations


def _get_rhofunc(rhomode):
    """
    Mapping from rhomode to rhofunc.
    """
    rhomode2func = {
        'frac': _rhoindex,
        'num': _rhoindex_num,
        'denom': _rhoindex_denom,
        'means': _rhoindex_means
        }
    return rhomode2func.get(rhomode, None)


def _get_rhomodesfx(rhomode):
    """
    Mapping from rhomode to rhomodesfx.
    """
    rhomode2sfx = {
        'frac': '',
        'num': '-rhoindex_num',
        'denom': '-rhoindex_denom',
        'means': '-rhoindex_means'
        }
    return rhomode2sfx.get(rhomode, None)


def _rhoindex(cmat, targetset):
    """
    Compute metastability index for given target set.
    """
    # Calculates the metastability index from prob cmat and target set
    nbins = len(cmat)
    tcomp = list(set(range(nbins)) - set(targetset))
    tset = cmat[:, targetset]
    probAB = min([1, np.max(tset[targetset])])
    probiM = min([1, min(np.max(tset[tcomp], axis=1))])
    rhoM = probAB / probiM
    return rhoM


def _rhoindex_num(cmat, targetset):
    """
    Compute metastability index for given target set.
    """
    # Calculates the metastability index from prob cmat and target set
    nbins = len(cmat)
    tcomp = list(set(range(nbins)) - set(targetset))
    tset = cmat[:, targetset]
    probAB = min([1, np.max(tset[targetset])])
    rhoM = probAB
    return rhoM


def _rhoindex_denom(cmat, targetset):
    """
    Compute metastability index for given target set.
    """
    # Calculates the metastability index from prob cmat and target set
    nbins = len(cmat)
    tcomp = list(set(range(nbins)) - set(targetset))
    tset = cmat[:, targetset]
    probiM = min([1, min(np.max(tset[tcomp], axis=1))])
    rhoM = 1.0 / probiM
    return rhoM


def _rhoindex_means(cmat, targetset):
    """
    Compute metastability index for given target set.
    """
    # Calculates the metastability index from prob cmat and target set
    nbins = len(cmat)
    tcomp = list(set(range(nbins)) - set(targetset))
    tset = cmat[:, targetset]
    probAB = min([1, np.mean(np.max(tset[targetset], axis=1))])
    probiM = min([1, np.mean(np.max(tset[tcomp], axis=1))])
    rhoM = probAB / probiM
    return rhoM


def _calc_qijk(mmat, i, j):
    """
    Compute committor function q_{ij}(k), given MFPT.
    Coordinates i, j as indices.
    q_ij(k) = (m_ki - m_kj + m_ij) / (m_ij + m_ji) ,
    where m_ii is set to 0.
    """
    mmat2 = mmat.copy()
    mmat2 -= np.diag(np.diag(mmat2))
    qij = (mmat2[:, i] - mmat2[:, j] + mmat2[i, j]) / \
            (mmat2[i, j] + mmat2[j, i])
    return qij


def _calc_qAi(mmat, targetset):
    """
    Compute attractor membership function q_A(i).
    """
    ntarget = len(targetset)
    # qAB(i)
    qABi = {}
    for target in targetset:
        complement = set(targetset) - set([target])
        for source in complement:
            qABi[source, target] = _calc_qijk(mmat, source, target)
    # c_j
    coeff = np.zeros((ntarget, ntarget - 1))
    st = time()
    for source in targetset:
        qABivalues = np.ones((ntarget - 1, ntarget - 1))
        setDiff = list(set(targetset) - set([source]))
        for tar1 in setDiff:
            for tar2 in setDiff:
                qABivalues[setDiff.index(tar1), setDiff.index(tar2)] = \
                                qABi[source, tar2][tar1]
        coeff[list(targetset).index(source)] = \
                        solve(qABivalues, np.ones(ntarget - 1))
    # qA(i)
    qAi = {i: 0 for i in targetset}
    for source in targetset:
        setDiff = list(set(targetset) - set([source]))
        for target in setDiff:
            qAi[source] += (qABi[source, target]) * \
                           coeff[list(targetset).index(source),
                               setDiff.index(target)]
        # source is target and vivecersa in the committor (1-qABi)
        qAi[source] = 1 - qAi[source]
    return qAi


def _calc_qAi_exact(fmat, targetset):
    """
    Compute attractor membership function q_A(i).
    """
    fmat = fmat - np.diag(np.diag(fmat))
    fmat /= np.sum(fmat)
    lmat = fmat.copy()
    for i in range(len(fmat)):
        lmat[i] /= np.sum(lmat[i])
    for i in range(len(fmat)):
        lmat[i, i] -= 1.0
    ntarget = len(targetset)
    tsetc = np.sort(list(set(np.arange(len(fmat))) - set(targetset)))
    qAi = {}
    for i, t in enumerate(targetset):
        thisqAi = np.zeros(len(fmat))
        thisqAi[t] = 1.0
        thisqAi[tsetc] = solve(lmat[tsetc][:, tsetc], -lmat[tsetc, t])
        qAi[t] = thisqAi
    return qAi


def _calc_qAi_sort(datadir, targetset, norm='raw'):
    """
    Compute committor probabilities q_A(i), as an arrays with rows
    corresponding to targets in sorted (increasing) order.
    """
    _, mmat, _, _ = _get_arrays(datadir, norm=norm)
    qAi = _calc_qAi(mmat, targetset)
    tset = np.sort(targetset)
    return np.array([qAi[t] for t in tset])


def _calc_qAi_sort_exact(datadir, targetset, norm='raw'):
    """
    Compute committor probabilities q_A(i), as an arrays with rows
    corresponding to targets in sorted (increasing) order.
    """
    fmat, _, _, _ = _get_arrays(datadir, norm=norm)
    qAi = _calc_qAi_exact(fmat, targetset)
    tset = np.sort(targetset)
    return np.array([qAi[t] for t in tset])

# Tset-based computations


def _calc_TargetEffLaplacian(datadir, tset, norm='raw'):
    """
    Compute effective Laplacian between targets.
    """
    fmat, mmat, cmat, mappingdata = _get_arrays(datadir, norm=norm)
    mapping = mappingdata[0]
    tset = np.sort(np.array(tset))
    ntarget = len(tset)
    # Create transition matrix pmat
    fmat -= np.diag(np.diag(fmat))
    pivec = np.sum(fmat, axis=0)
    pmat = np.array([v / pivec[i] for i, v in enumerate(fmat)])
    pivec /= np.sum(pivec)
    lmat = pmat - np.diag(np.ones_like(pivec))
    lab = np.zeros((ntarget, ntarget))
    for aind in range(ntarget):
        for bind in range(ntarget):
            if aind == bind:
                continue
            qab = _calc_qijk(mmat, tset[aind], tset[bind])
            fijab = np.outer(pivec * (1.0 - qab), qab) * lmat
            fijAB = fijab - fijab.T
            fijAB[fijAB < 0.0] = 0.0
            lab[aind, bind] = np.sum(fijAB[tset[aind]])
    return lab


def _calc_TargetEffLaplacian_20160802(datadir, tset, norm='raw'):
    """
    Compute effective Laplacian between targets.
    Based on computations / formula laid out by Enrico in:
        J Chem Phys 145, 024102 (2016).
    Note: L_ab is the symmetric Laplacian matrix.
    """
    fmat, mmat, cmat, mappingdata = _get_arrays(datadir, norm=norm)
    fmat -= np.diag(np.diag(fmat))
    fmat /= np.sum(fmat)
    mapping = mappingdata[0]
    tset = np.sort(np.array(tset))
    ntarget = len(tset)
    pivec = np.sum(fmat, axis=0) / np.sum(fmat)
    qAi = _get_TargetCommittor(datadir, tset, norm=norm)
    pmat = np.array([v / pivec[i] for i, v in enumerate(fmat)])
    lmat = pmat - np.diag(np.ones_like(pivec))
    lmat2 = lmat[:, tset]
    qAi2 = qAi * np.array([list(pivec)] * ntarget)
    pivec2 = np.dot(qAi, pivec)
    # Asymmetric k_ab transition matrix
    #lab = np.dot(qAi2, lmat2) / np.array([list(pivec2)] * ntarget).T
    lab = np.dot(qAi2, lmat2)
    return lab


def _calc_TargetBinLaplacian_20160802(datadir, tset, norm='raw',
                membership='soft', matnorm='average'):
    """
    Compute effective Laplacian between targets, based on simple binning
     operation using hitting probabilities q_alpha.
    membership: Use 'soft' or 'hard' partitioning
    matnorm: Use 'average' to output average interaction per pixel^2,
             else 'sum'.
    Note: L_ab is the symmetric Laplacian matrix.
    """
    fmat, _, _, mappingdata = _get_arrays(datadir, norm=norm)
    mapping = mappingdata[0]
    fmat -= np.diag(np.diag(fmat))
    fmat /= np.sum(fmat)
    tset = np.sort(np.array(tset))
    ntarget = len(tset)
    qAi = _get_TargetCommittor(datadir, tset, norm=norm)
    membership = qAi.copy()
    if membership == 'hard':
        for i in range(qAi.shape[1]):
            membership[:, i] = 0.0
            membership[np.argmax(qAi[:, i]), i] = 1.0
    if matnorm == 'average':
        for i in range(len(membership)):
            membership[i] /= np.sum(membership[i])
    lab1 = np.dot(membership, np.dot(fmat, membership.T))
    lab = np.array([[np.dot(np.dot(membership[i], fmat), membership[j])
                     for j in range(ntarget)]
                    for i in range(ntarget)])
    if not np.allclose(lab, lab1):
        print 'Not correct!'
    del fmat
    return lab


def _tset_mergeIslands(tset, membership, mappingdata,
                       nmapmax=100, cutoffsize=10):
    """
    Create hard partitioning from padded membership with merged islands.
    Returns padded membership function.
    """
    mapping = mappingdata[0]
    qAip = np.array([np.argmax(vec) if np.sum(vec) > 0.0 else -1
                     for vec in membership.T])
    # Regularize islands
    # Get mapped qAip
    qAip_old = qAip.copy()
    nmap = 0
    while nmap < nmapmax:
        nmap += 1
        qAip_new = _mapqAiIslands(qAip_old,
                        cutoffsize, mapping, tset)
        if np.allclose(qAip_new, qAip_old):
            qAip2 = qAip_new
            break
        else:
            qAip_old = qAip_new.copy()
    # Create membership function
    bounds2 = np.abs(np.array([-2] + list(qAip2)) -
                     np.array(list(qAip2) + [-2])) > 0
    bounds2 = np.nonzero(bounds2)[0]
    sizes2 = bounds2[1:] - bounds2[:-1]
    npieces2 = len(sizes2)
    lims2 = [bounds2[i:i + 2] for i in range(npieces2)]
    pieceids2 = [int(qAip2[l[0]]) for l in lims2]
    membership = np.zeros((np.max(pieceids2) + 1, len(qAip)))
    for ind, (st, en) in zip(pieceids2, lims2):
        if ind >= 0:
            membership[ind, st:en] = 1.0
    return membership


def _get_limsIds_fromPaddedSplitMembership(membership):
    """
    Get partition region definitions from padded membership.
    Assumes disjoint partitions are split.
    """
    qAip2 = np.array([np.argmax(vec) if np.max(vec) > 0.0 else -1
                      for vec in membership.T])
    bounds2 = np.abs(np.array([-2] + list(qAip2)) -
                     np.array(list(qAip2) + [-2])) > 0
    bounds2 = np.nonzero(bounds2)[0]
    sizes2 = bounds2[1:] - bounds2[:-1]
    npieces2 = len(sizes2)
    lims2 = [bounds2[i:i + 2] for i in range(npieces2)]
    pieceids2 = [int(qAip2[l[0]]) for l in lims2]
    return lims2, pieceids2


def _tset_splitDisjointPartitions(membership):
    """
    Split disjoint hard partitions from padded membership function.
    """
    newmembershipvecs = []
    for ivec, vec in enumerate(membership):
        vecpad = np.round(np.array([0] + list(vec) + [0]))
        diffvec = vecpad[1:] - vecpad[:-1]
        starts = np.nonzero(diffvec > 0.0)[0]
        ends = np.nonzero(diffvec < 0.0)[0]
        if len(starts) > 1:
            print 'Membership row', ivec, 'split into', len(starts), '.'
            print 'starts:', starts
            print 'ends:', ends
        for st, en in zip(starts, ends):
            a = np.zeros(len(vec))
            a[st:en] = 1.0
            newmembershipvecs.append(a)
    membership2 = np.array(newmembershipvecs)
    starts = [np.min(np.where(vec == 1.0)) for vec in membership2]
    membership2 = membership2[np.argsort(starts)]
    return membership2


def _get_mappedPaddedMembership(pars, tset, norm='raw',
                merge=False, cutoffsize=10, nmapmax=100, split=False):
    """
    Compute membership function for hard partitions, based on simple binning
     operation using hitting probabilities q_alpha.
    matnorm: Use 'average' to output average interaction per pixel^2.
             Use 'sqrt' to set weight = sqrt(number of pixels).
             Otherwise, use 'sum' to simply sum up all interaction counts.
    merge: Used only for membertype = 'hard'
           Merge islands that are smaller than cutoffsize pixels.
    split: Used only for membertype = 'hard'
           Split disconnected partitions if True.
    Note: L_ab is the symmetric Laplacian matrix.
    """
    tsetbeta = pars['tsetbeta']
    fmatbeta = pars['fmatbeta']
    tpars = copy.deepcopy(pars)
    tpars['beta'] = tsetbeta
    tsetdatadir = _get_runbinarydir(tpars)
    tsetmapping, _ = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    tpars['beta'] = fmatbeta
    fmatdatadir = _get_runbinarydir(tpars)
    fmatmapping, _ = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    mappedtset = tsetmapping[tset]
    tset2 = [list(fmatmapping).index(t) for t in mappedtset]
    tset2fmatmapping = np.array([list(fmatmapping).index(t)
                    for t in tsetmapping])
    fmat, _, _, mappingdata = _get_arrays(fmatdatadir, norm=norm)
    mapping = mappingdata[0]
    fmat -= np.diag(np.diag(fmat))
    #fmat /= np.sum(fmat)
    tset = np.sort(np.array(tset))
    qAi = _get_TargetCommittor(tsetdatadir, tset, norm=norm)
    qAip = np.zeros((len(qAi), len(fmatmapping)))
    qAip[:, tset2fmatmapping] = qAi
    membership = qAip.copy()
    for i in range(qAip.shape[1]):
        membership[:, i] = 0.0
        membership[np.argmax(qAip[:, i]), i] = 1.0
    # Pad membership array
    membership2 = np.zeros((len(membership), mappingdata[1]))
    membership2[:, mapping] = membership
    if merge:
        membership2 = _tset_mergeIslands(tset, membership2, mappingdata,
                           nmapmax=nmapmax, cutoffsize=cutoffsize)
    if split:
        membership2 = _tset_splitDisjointPartitions(membership2)
    return membership2


def _calc_TargetBinLaplacian_20160802_mixbeta(pars, tset, norm='raw',
                membertype='soft', matnorm='average', merge=False,
                cutoffsize=10, nmapmax=100, split=False, getmemb=False):
    """
    Compute effective Laplacian between targets, based on simple binning
     operation using hitting probabilities q_alpha.
    membertype: Use 'soft' or 'hard' partitioning
    matnorm: Use 'average' to output average interaction per pixel^2.
             Use 'sqrt' to set weight = sqrt(number of pixels).
             Otherwise, use 'sum' to simply sum up all interaction counts.
    merge: Used only for membertype = 'hard'
           Merge islands that are smaller than cutoffsize pixels.
    split: Used only for membertype = 'hard'
           Split disconnected partitions if True.
    getmemb: Get padded membership matrix also.
    Note: L_ab is the symmetric Laplacian matrix.
    """
    tsetbeta = pars['tsetbeta']
    fmatbeta = pars['fmatbeta']
    tpars = copy.deepcopy(pars)
    tpars['beta'] = tsetbeta
    tsetdatadir = _get_runbinarydir(tpars)
    tsetmapping, _ = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    tpars['beta'] = fmatbeta
    fmatdatadir = _get_runbinarydir(tpars)
    fmatmapping, _ = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    mappedtset = tsetmapping[tset]
    tset2 = [list(fmatmapping).index(t) for t in mappedtset]
    tset2fmatmapping = np.array([list(fmatmapping).index(t)
                    for t in tsetmapping])
    fmat, _, _, mappingdata = _get_arrays(fmatdatadir, norm=norm)
    mapping = mappingdata[0]
    fmat -= np.diag(np.diag(fmat))
    #fmat /= np.sum(fmat)
    tset = np.sort(np.array(tset))
    qAi = _get_TargetCommittor(tsetdatadir, tset, norm=norm)
    qAip = np.zeros((len(qAi), len(fmatmapping)))
    qAip[:, tset2fmatmapping] = qAi
    membership = qAip.copy()
    if membertype == 'hard':
        for i in range(qAip.shape[1]):
            membership[:, i] = 0.0
            membership[np.argmax(qAip[:, i]), i] = 1.0
        if merge or split:
            # Pad membership array
            membership2 = np.zeros((len(membership), mappingdata[1]))
            membership2[:, mapping] = membership
            if merge:
                membership2 = _tset_mergeIslands(tset, membership2, mappingdata,
                                   nmapmax=nmapmax, cutoffsize=cutoffsize)
            if split:
                membership2 = _tset_splitDisjointPartitions(membership2)
            # Unpad membership array
            membership = membership2[:, mapping]
    npartitions = len(membership)
    if matnorm == 'average':
        for i in range(len(membership)):
            membership[i] /= np.sum(membership[i])
    elif matnorm == 'sqrt':
        for i in range(len(membership)):
            membership[i] /= np.sqrt(np.sum(membership[i]))
    lab1 = np.dot(membership, np.dot(fmat, membership.T))
    lab = np.array([[np.dot(np.dot(membership[i], fmat), membership[j])
                     for j in range(npartitions)]
                    for i in range(npartitions)])
    if not np.allclose(lab, lab1):
        print 'Not correct!'
    del fmat
    if getmemb:
        return lab, membership2
    else:
        return lab


def _calc_TargetEffLaplacian_20160829(pars, tset, norm='raw'):
    """
    Compute effective Laplacian between targets.
    Based on computations / formula laid out by Enrico in:
        J Chem Phys 145, 024102 (2016).
    Note: L_ab is the symmetric Laplacian matrix.
          Computed using fmat at beta = 1.0
    """
    tpars = copy.deepcopy(pars)
    beta = pars['beta']
    tpars['beta'] = 1.0
    datadir = _get_runbinarydir(tpars)
    fmat, _, _, mappingdata = _get_arrays(datadir, norm=norm)
    fmat[np.isnan(fmat)] = 0.0
    fmat -= np.diag(np.diag(fmat))
    fmat /= np.sum(fmat)
    mapping, nbins = mappingdata
    # Map onto full array
    fmatfull = plu._build_fullarray(fmat, mappingdata, 0.0)
    tset = np.sort(np.array(tset))
    ntarget = len(tset)
    pivec = np.sum(fmatfull, axis=0) / np.sum(fmatfull)
    print np.sum(pivec), np.sum(pivec == 0.0)
    tpars['beta'] = beta
    datadir = _get_runbinarydir(tpars)
    qAi = _get_TargetCommittor(datadir, tset, norm=norm)
    mappingdata = _get_mappingdata(datadir, norm=norm)
    print np.sum(qAi), qAi.shape
    qAifull = np.array([plu._build_fullvector(q, mappingdata, 0.0)
                    for q in qAi])
    print np.sum(qAifull), qAifull.shape
    pmat = np.array([v / pivec[i] if pivec[i] > 0.0 else v * 0.0
                     for i, v in enumerate(fmatfull)])
    lmat = pmat - np.diag(np.ones_like(pivec))
    lmat2 = lmat[:, mapping[tset]]
    print np.sum(lmat2)
    qAi2 = qAifull * np.array([list(pivec)] * ntarget)
    #pivec2 = np.dot(qAi, pivec)
    # Asymmetric k_ab transition matrix
    #lab = np.dot(qAi2, lmat2) / np.array([list(pivec2)] * ntarget).T
    lab = np.dot(qAi2, lmat2)
    return lab


def _calc_TargetEffLaplacian_interchr_20160802(pars):
    """
    Get effective Laplacian between targets: inter-chromosomal case.
    Requires pars: chrfullname1/2, ntarget1/2, beta, norm.
    Formula for a in c1, b in c2:
        l_{a,b} = sum_{i, j in c1 union c2} {q_a(i) pi_i L_{i,j} q_b(j)}
    """
    tpars = copy.deepcopy(pars)
    chrfullname1 = pars['chrfullname1']
    chrfullname2 = pars['chrfullname2']
    ntarget1 = pars['ntarget1']
    ntarget2 = pars['ntarget2']
    norm = pars['norm']
    # Get tsets, committors
    tpars['chrfullname'] = chrfullname1
    mappedtset1 = _get_tsetReference_20180802(tpars, ntarget1)[0]
    tpars['chrref'] = chrfullname1
    mapping, _ = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    tset1 = np.array([list(mapping).index(t) for t in mappedtset1])
    qAc1 = _get_TargetCommittor(_get_runbinarydir(tpars), tset1,
                    norm=norm)
    tpars['chrfullname'] = chrfullname2
    mappedtset2 = _get_tsetReference_20180802(tpars, ntarget2)[0]
    tpars['chrref'] = chrfullname2
    mapping, _ = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    tset2 = np.array([list(mapping).index(t) for t in mappedtset2])
    qBc2 = _get_TargetCommittor(_get_runbinarydir(tpars), tset2,
                    norm=norm)
    # Get trans-fij, targetmembership
    tpars['chrref'] = chrfullname1
    fmat1, _, _, md1 = _get_arrays(_get_runbinarydir(tpars), norm=norm)
    tpars['chrref'] = chrfullname2
    fmat2, _, _, md2 = _get_arrays(_get_runbinarydir(tpars), norm=norm)
    pars['chrref'] = '_'.join([chrfullname1, chrfullname2])
    fmat12, mdp1, mdp2 = _get_fmatmap_inter(pars)
    targetmembership = _targetmembership_softpartition_2chr(
                    qAc1, qBc2, md1, md2)
    # Compute lac1bc2
    ## Trans-lij
    nbins1, nbins2 = md1[1], md2[1]
    fillmat1 = plu._build_fullarray_inter(fmat1, md1, md1, np.nan)
    fillmat12 = plu._build_fullarray_inter(fmat12, mdp1, mdp2, np.nan)
    fillmat2 = plu._build_fullarray_inter(fmat2, md2, md2, np.nan)
    fmatcombined = np.array(np.bmat([[fillmat1, fillmat12],
                 [fillmat12.T, fillmat2]]))
    del fillmat1, fillmat12, fillmat2
    fmatcombined2 = fmatcombined.copy()
    fmatcombined2[np.isnan(fmatcombined)] = 0.0
    del fmatcombined
    pivec = np.sum(fmatcombined2, axis=1)
    pmatcombined = np.array([v / pivec[i] if pivec[i] > 0.0 else 0.0 * v
                    for i, v in enumerate(fmatcombined2)])
    pivec /= np.sum(pivec)
    lmatcombined = pmatcombined - np.diag(np.ones_like(pivec))
    del fmatcombined2, pmatcombined
    ## lac1bc2
    lac1bc2 = np.zeros((ntarget1, ntarget2))
    for a1 in range(ntarget1):
        for b2 in range(ntarget2):
            v1 = targetmembership[a1] * pivec
            v2 = targetmembership[b2 + ntarget1]
            filtermat = np.outer(v1, v2)
            lac1bc2[a1, b2] = np.sum(filtermat * lmatcombined)
    return lac1bc2


def _calc_TargetEffLaplacian_interchr_20160829(pars):
    """
    Get effective Laplacian between targets: inter-chromosomal case.
    Requires pars: chrfullname1/2, ntarget1/2, beta, norm.
    Note: Uses fmat from beta = 1.0
    Formula for a in c1, b in c2:
        l_{a,b} = sum_{i, j in c1 union c2} {q_a(i) pi_i L_{i,j} q_b(j)}
    """
    tpars = copy.deepcopy(pars)
    chrfullname1 = pars['chrfullname1']
    chrfullname2 = pars['chrfullname2']
    ntarget1 = pars['ntarget1']
    ntarget2 = pars['ntarget2']
    norm = pars['norm']
    beta = pars['beta']
    # Get tsets, committors
    tpars['chrfullname'] = chrfullname1
    mappedtset1 = _get_tsetReference_20180802(tpars, ntarget1)[0]
    tpars['chrref'] = chrfullname1
    mp1 = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    tset1 = np.array([list(mp1[0]).index(t) for t in mappedtset1])
    qAc1 = _get_TargetCommittor(_get_runbinarydir(tpars), tset1,
                    norm=norm)
    tpars['chrfullname'] = chrfullname2
    mappedtset2 = _get_tsetReference_20180802(tpars, ntarget2)[0]
    tpars['chrref'] = chrfullname2
    mp2 = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    tset2 = np.array([list(mp2[0]).index(t) for t in mappedtset2])
    qBc2 = _get_TargetCommittor(_get_runbinarydir(tpars), tset2,
                    norm=norm)
    # Get trans-fij, targetmembership
    tpars['beta'] = 1.0
    tpars['chrref'] = chrfullname1
    fmat1, _, _, md1 = _get_arrays(_get_runbinarydir(tpars), norm=norm)
    tpars['chrref'] = chrfullname2
    fmat2, _, _, md2 = _get_arrays(_get_runbinarydir(tpars), norm=norm)
    pars['chrref'] = '_'.join([chrfullname1, chrfullname2])
    fmat12, mdp1, mdp2 = _get_fmatmap_inter(pars)
    targetmembership = _targetmembership_softpartition_2chr(
                    qAc1, qBc2, mp1, mp2)
    # Compute lac1bc2
    ## Trans-lij
    nbins1, nbins2 = md1[1], md2[1]
    fillmat1 = plu._build_fullarray_inter(fmat1, md1, md1, np.nan)
    fillmat12 = plu._build_fullarray_inter(fmat12, mdp1, mdp2, np.nan)
    fillmat2 = plu._build_fullarray_inter(fmat2, md2, md2, np.nan)
    fmatcombined = np.array(np.bmat([[fillmat1, fillmat12],
                 [fillmat12.T, fillmat2]]))
    del fillmat1, fillmat12, fillmat2
    fmatcombined2 = fmatcombined.copy()
    fmatcombined2[np.isnan(fmatcombined)] = 0.0
    del fmatcombined
    pivec = np.sum(fmatcombined2, axis=1)
    pmatcombined = np.array([v / pivec[i] if pivec[i] > 0.0 else 0.0 * v
                    for i, v in enumerate(fmatcombined2)])
    pivec /= np.sum(pivec)
    lmatcombined = pmatcombined - np.diag(np.ones_like(pivec))
    del fmatcombined2, pmatcombined
    ## lac1bc2
    lac1bc2 = np.zeros((ntarget1, ntarget2))
    for a1 in range(ntarget1):
        for b2 in range(ntarget2):
            v1 = targetmembership[a1] * pivec
            v2 = targetmembership[b2 + ntarget1]
            filtermat = np.outer(v1, v2)
            lac1bc2[a1, b2] = np.sum(filtermat * lmatcombined)
    return lac1bc2


def _calc_TargetEffLaplacian_interchr_20160830(pars):
    """
    Get effective Laplacian between targets: inter-chromosomal case.
    Requires pars: chrfullname1/2, ntarget1/2, beta, norm.
    Formula for a in c1, b in c2:
        l_{a,b} = sum_{i in c1} {q_a(i) pi_i L_{i,b}}
    """
    tpars = copy.deepcopy(pars)
    chrfullname1 = pars['chrfullname1']
    chrfullname2 = pars['chrfullname2']
    ntarget1 = pars['ntarget1']
    ntarget2 = pars['ntarget2']
    norm = pars['norm']
    # Get tsets, committors
    tpars['chrfullname'] = chrfullname1
    mappedtset1 = _get_tsetReference_20180802(tpars, ntarget1)[0]
    tpars['chrref'] = chrfullname1
    mapping, _ = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    tset1 = np.array([list(mapping).index(t) for t in mappedtset1])
    qAc1 = _get_TargetCommittor(_get_runbinarydir(tpars), tset1,
                    norm=norm)
    tpars['chrfullname'] = chrfullname2
    mappedtset2 = _get_tsetReference_20180802(tpars, ntarget2)[0]
    tpars['chrref'] = chrfullname2
    mapping, _ = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    tset2 = np.array([list(mapping).index(t) for t in mappedtset2])
    qBc2 = _get_TargetCommittor(_get_runbinarydir(tpars), tset2,
                    norm=norm)
    # Get trans-fij, targetmembership
    tpars['chrref'] = chrfullname1
    fmat1, _, _, md1 = _get_arrays(_get_runbinarydir(tpars), norm=norm)
    tpars['chrref'] = chrfullname2
    fmat2, _, _, md2 = _get_arrays(_get_runbinarydir(tpars), norm=norm)
    pars['chrref'] = '_'.join([chrfullname1, chrfullname2])
    fmat12, mdp1, mdp2 = _get_fmatmap_inter(pars)
    targetmembership = _targetmembership_softpartition_2chr(
                    qAc1, qBc2, md1, md2)
    # Compute lac1bc2
    ## Trans-lij
    nbins1, nbins2 = md1[1], md2[1]
    fillmat1 = plu._build_fullarray_inter(fmat1, md1, md1, np.nan)
    fillmat12 = plu._build_fullarray_inter(fmat12, mdp1, mdp2, np.nan)
    fillmat2 = plu._build_fullarray_inter(fmat2, md2, md2, np.nan)
    fmatcombined = np.array(np.bmat([[fillmat1, fillmat12],
                 [fillmat12.T, fillmat2]]))
    del fillmat1, fillmat12, fillmat2
    fmatcombined2 = fmatcombined.copy()
    fmatcombined2[np.isnan(fmatcombined)] = 0.0
    del fmatcombined
    pivec = np.sum(fmatcombined2, axis=1)
    pmatcombined = np.array([v / pivec[i] if pivec[i] > 0.0 else 0.0 * v
                    for i, v in enumerate(fmatcombined2)])
    pivec /= np.sum(pivec)
    lmatcombined = pmatcombined - np.diag(np.ones_like(pivec))
    del fmatcombined2, pmatcombined
    ## lac1bc2
    lac1bc2 = np.zeros((ntarget1, ntarget2))
    for a1 in range(ntarget1):
        #for b2 in range(ntarget2):
        v1 = targetmembership[a1] * pivec
        lac1bc2[a1] = np.dot(v1, lmatcombined[:, mappedtset2 + nbins1])
        #v2 = targetmembership[b2 + ntarget1]
        #filtermat = np.outer(v1, v2)
        #lac1bc2[a1, b2] = np.sum(filtermat * lmatcombined)
    return lac1bc2


def _targetmembership_hardpartition(pars, qAi, mappingdata):
    """
    Compute target membership function.
    Hard partitioning: assign each locus to the highest-committor target.
    """
    ntarget = len(qAi)
    mapping, nbins = mappingdata
    targetmembership = np.zeros((ntarget, nbins))
    for i in range(nbins):
        if i not in mapping:
            continue
        ind = np.where(mapping == i)
        targetmembership[np.argmax(qAi[:, ind]), i] = 1.0
    return targetmembership


def _targetmembership_softpartition(pars, qAi, mappingdata):
    """
    Compute target membership function.
    Soft partitioning: For each target, weight the contribution of each locus
                       by the committor function.
    """
    ntarget = len(qAi)
    mapping, nbins = mappingdata
    targetmembership = np.zeros((ntarget, nbins))
    targetmembership[:, mapping] = qAi.copy()
    return targetmembership


def _targetmembership_splitcenterpartition(pars, tset, mappingdata):
    """
    Compute target membership function.
    Split-center partitioning: assign each locus to the nearest target
                               along genome.
    """
    # Force sorting
    tset2 = np.sort(tset)
    ntarget = len(tset)
    mapping, nbins = mappingdata
    targetmembership = np.zeros((ntarget, nbins))
    for i in range(nbins):
        if i not in mapping:
            continue
        ind = np.where(mapping == i)
        distances = np.abs(ind - tset2)
        targetmembership[np.argmin(distances), i] = 1.0
    return targetmembership


def _targetmembership_hardpartition_2chr(qAc1, qBc2, md1, md2):
    """
    Compute target membership function, indices mapped over 2 chromosomes.
    Hard partitioning: assign each locus to the highest-committor target.
    """
    ntarget1, ntarget2 = len(qAc1), len(qBc2)
    nbins1, nbins2 = md1[1], md2[1]
    nbins12 = nbins1 + nbins2
    ntarget12 = ntarget1 + ntarget2
    targetmembership = np.zeros((ntarget12, nbins12))
    for i in range(nbins1):
        if i not in md1[0]:
            continue
        ind = np.where(md1[0] == i)
        targetmembership[np.argmax(qAc1[:, ind]), i] = 1.0
    for i in range(nbins2):
        if i not in md2[0]:
            continue
        ind = np.where(md2[0] == i)
        targetmembership[np.argmax(qBc2[:, ind]) + ntarget1, i + nbins1] = 1.0
    return targetmembership


def _targetmembership_softpartition_2chr(qAc1, qBc2, md1, md2):
    """
    Compute target membership function.
    Soft partitioning: For each target, weight the contribution of each locus
                       by the committor function.
    """
    ntarget1, ntarget2 = len(qAc1), len(qBc2)
    nbins1, nbins2 = md1[1], md2[1]
    nbins12 = nbins1 + nbins2
    ntarget12 = ntarget1 + ntarget2
    targetmembership = np.zeros((ntarget12, nbins12))
    targetmembership[:ntarget1, md1[0]] = qAc1.copy()
    targetmembership[ntarget1:, md2[0] + nbins1] = qBc2.copy()
    return targetmembership


def _mapqAiIslands(qAip, cutoffsize, mapping, tset):
    """
    Map islands smaller than cutoffsize pixels on chromosome to suitable
    partitions.
    Input:
        - qAip: Vector of partition assignments.
                Set to -1 for pixels not in fmat.
        - cutoffsize: Cutoff size for islands, in pixels.
    """
    ntarget = len(tset)
    # Get splitting
    bounds = np.abs(np.array([-2] + list(qAip)) -
                    np.array(list(qAip) + [-2])) > 0
    bounds = np.nonzero(bounds)[0]
    sizes = bounds[1:] - bounds[:-1]
    npieces = len(sizes)
    lims = [bounds[i:i + 2] for i in range(npieces)]
    pieceids = [int(qAip[l[0]]) for l in lims]
    # Identify islands
    islandmask = (sizes <= cutoffsize)
    chunkmask = (sizes > cutoffsize)
    islandonlyparts = []
    for i in range(ntarget):
        if i not in np.array(pieceids)[chunkmask]:
            islandonlyparts.append(i)
    # Reassign islands
    qAip2 = qAip.copy()
    for i in range(npieces):
        if chunkmask[i] or pieceids[i] in islandonlyparts:
            continue
        lo, hi = lims[i]
        # Get piece id and priority levels of lo and hi ends
        shift = 1
        ilo = i - shift
        ihi = i + shift
        lo_id = -2 if i == 0 else pieceids[ilo]
        lo_priority = -2 if i == 0 else (chunkmask[ilo] * 1
                        if lo_id > -1 else -1)
        hi_id = -2 if i == (npieces - 1) else pieceids[ihi]
        hi_priority = -2 if i == (npieces - 1) else (chunkmask[ihi] * 1
                        if hi_id > -1 else -1)
        # Check if need to search next nearest neighbors
        if lo_id == hi_id and lo_id == -1:
            shift = 2
            ilo = i - shift
            ihi = i + shift
            lo_id = -2 if i <= 1 else pieceids[ilo]
            hi_id = -2 if i >= npieces - 2 else pieceids[ihi]
            lo_mt = np.inf if lo_id < 0 else mapping[tset[lo_id]]
            hi_mt = np.inf if hi_id < 0 else mapping[tset[hi_id]]
            this_id = lo_id if np.abs(lo - lo_mt) < np.abs(hi_mt - hi) \
                            else hi_id
            #print 'Case 1 ',
        # 1) If embedded in single partition...
        elif lo_id == hi_id:
            this_id = lo_id
            #print 'Case 2 ',
        # Compare priority
        elif lo_priority > hi_priority:
            this_id = lo_id
            #print 'Case 3 ',
        elif lo_priority < hi_priority:
            this_id = hi_id
            #print 'Case 4 ',
        else:
            # Find closer target
            lo_mt = mapping[tset[lo_id]]
            hi_mt = mapping[tset[hi_id]]
            this_id = lo_id if np.abs(lo - lo_mt) < np.abs(hi_mt - hi) \
                            else hi_id
            #print 'Case 5 ',
        #print 'piece %i, qAip2[%i:%i] set to %i' % (i, lo, hi, this_id)
        qAip2[lo:hi] = this_id
    return qAip2


##########################################################################
# Specialized analysis functions
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


##


##############################################################
# Target set optimization


def _trynewtarget_construct(cmat, targetset, rhofunc=_rhoindex):
    nbins = len(cmat)
    tset = list(targetset)
    scores = []
    ntarget = len(targetset) + 1
    for trial in range(nbins):
        if trial in targetset:
            scores.append(np.nan)
            continue
        trialset = tset + [trial]
        scores.append(rhofunc(cmat, np.array(trialset)))
    scores = np.array(scores)
    minarg = np.isfinite(scores).nonzero()[0][
                np.argmin(scores[np.isfinite(scores)])]
    return minarg, scores[minarg]


def _choosesteppers(initial, pstep, minstep=1):
    """
    Utility for MC move chooser: Select points to step.
    Note: Requires initial to be a 1D list / array.
    """
    npts = len(initial)
    pstep2 = max(float(minstep) / npts, pstep)
    while True:
        trialstep = np.random.uniform(size=npts) < pstep2
        nstep = np.sum(trialstep)
        if nstep > 0:
            break
    stepfrom = list(np.array(initial)[trialstep])
    stayat = list(np.array(initial)[np.array(1 - trialstep, dtype='bool')])
    return stayat, stepfrom


def _trialstep_vectormap(stayat, stepfrom, allchoices, vectormap):
    """
    Utility for MC move chooser: Step from selected points,
    adding random points if overlap occurs.
    """
    npts = len(stayat) + len(stepfrom)
    stepto = np.array(vectormap)[np.array(stepfrom)]
    trialset = list(set().union(stayat, stepto))
    deficit = npts - len(trialset)
    if deficit > 0:
        choices = list(set(allchoices) - set(trialset))
        random.shuffle(choices)
        trialset = trialset + choices[:deficit]
    return trialset


def _trialstep_random(stayat, stepfrom, allchoices):
    """
    Utility for MC move chooser: Step from selected points,
    adding random points if overlap occurs.
    """
    npts = len(stayat) + len(stepfrom)
    choices = list(set(allchoices) - set(stayat) - set(stepfrom))
    random.shuffle(choices)
    stepto = choices[:len(stepfrom)]
    trialset = list(set().union(stayat, stepto))
    deficit = npts - len(trialset)
    if deficit > 0:
        choices = list(set(allchoices) - set(trialset))
        random.shuffle(choices)
        trialset = trialset + choices[:deficit]
    return trialset


def _stepping_gammamax(targetset, cmat, bestrho, besttset, gammamaxmap,
                allchoices, nstep, pstep, kT, minstep=1, rhofunc=_rhoindex):
    #print targetset
    for j in range(nstep):
        # Choose which ones to step: Require at least 1 step
        stayat, stepfrom = _choosesteppers(targetset,
                        pstep, minstep=minstep)
        # Trial step
        if not set(stepfrom).issubset(set(allchoices)):
            print len(cmat), stepfrom, stayat
        trialset = _trialstep_vectormap(stayat, stepfrom,
                        allchoices, gammamaxmap)
        # Metropolis criterion for deciding whether or not to move
        thisrho = rhofunc(cmat, targetset)
        trialrho = rhofunc(cmat, trialset)
        if _Metropolis_decision(thisrho, trialrho, kT):
            targetset = copy.deepcopy(trialset)
            if trialrho < bestrho:
                print 'Gamma move', j, 'good'
                bestrho = trialrho
                besttset = copy.deepcopy(trialset)
    return targetset, bestrho, besttset


def _stepping_random(targetset, cmat, bestrho, besttset,
                allchoices, nstep, pstep, kT, minstep=1, rhofunc=_rhoindex):
    for j in range(nstep):
        # Choose which ones to step: Require at least 1 step
        stayat, stepfrom = _choosesteppers(targetset,
                        pstep, minstep=1)
        # Trial step
        trialset = _trialstep_random(stayat, stepfrom, allchoices)
        thisrho = rhofunc(cmat, targetset)
        trialrho = rhofunc(cmat, trialset)
        if _Metropolis_decision(thisrho, trialrho, kT):
            targetset = copy.deepcopy(trialset)
            if trialrho < bestrho:
                print 'Random move', j, 'good'
                bestrho = trialrho
                besttset = copy.deepcopy(trialset)
    return targetset, bestrho, besttset


def _Metropolis_decision(initialscore, finalscore, kT):
    """
    Utility for MC decision based on Metropolis criterion.
    Accept if finalscore < initialscore, or with probability
    exp((initialscore - finalscore) / kT)...
    """
    return (finalscore < initialscore or
            np.random.uniform() < np.exp((initialscore - finalscore) / kT))


def run_ConstructMC_arbarray(fmat, cmat, datalabel, tsetdatadir,
                tsetdataprefix, steppars, ntargetmax=None, optimize2=False):
    """
    Run Construct-MC optimization for rho on a range of ntarget: [2, ntargetmax]
    """
    # Unpack parameters
    nstep_gammamax, pstep_gammamax, nstep_random, pstep_random, kT = steppars
    nbins = len(fmat)
    # Set ntargetmax, if None, to len(fmat) - 1
    if ntargetmax is None:
        ntargetmax = nbins - 1
    print 'ntargetmax =', ntargetmax
    rhodata = {}
    tsetdata = {}
    # Read cmat, fmat, mmat, mapping
    nbins = len(cmat)
    allchoices = range(nbins)
    plttitle = 'Arbitrary array'
    # Find gamma-maximizing mapping
    gammamaxmap = np.array([np.argmax(v) for v in cmat])
    print
    # Find seed target pair: Optimal from current data, or pivec-seeded
    if optimize2:
        st = time()
        targetset = [0, 1]
        bestrho = _rhoindex(cmat, targetset)
        for i in range(nbins):
            for j in range(i + 1, nbins):
                trialset = [i, j]
                thisrho = _rhoindex(cmat, trialset)
                if thisrho < bestrho:
                    bestrho = thisrho
                    targetset = trialset
        key = (datalabel, 2)
        rhodata[key] = bestrho
        tsetdata[key] = targetset
        en = time()
        print 2, ('(%.2e secs):' % (en - st)), bestrho
    else:
        st = time()
        rhodict0, tsetdict0 = _get_rhotsetdicts(tsetdatadir,
                        tsetdataprefix)
        key = (datalabel, 2)
        if key in rhodict0:
            print 'Load 2-target state from data'
            newrho = rhodict0[key]
            targetset = tsetdict0[key]
        else:
            print 'Seed 2-target state from pivec'
            pivec = np.sum(fmat, axis=1)
            targetset = [np.argmax(pivec)]
            newtarget, newrho = _trynewtarget_construct(cmat, targetset)
            targetset = targetset + [newtarget]
        del rhodict0, tsetdict0
        ### Start MC with candidate:
        bestrho = newrho
        besttset = copy.deepcopy(targetset)
        #### Follow cmat-maximizing trajectory
        ntarget = len(besttset)
        targetset, bestrho, besttset = _stepping_gammamax(targetset, cmat,
                bestrho, besttset, gammamaxmap, allchoices,
                nstep_gammamax, pstep_gammamax, kT, minstep=1)
        #### Random trials
        targetset, bestrho, besttset = _stepping_random(targetset,
                cmat, bestrho, besttset, allchoices,
                nstep_random, pstep_random, kT, minstep=1)
        ### Record rhodata, targetsetdata
        targetset = besttset
        rhodata[key] = bestrho
        tsetdata[key] = targetset
        en = time()
        print 2, ('(%.2e secs):' % (en - st)), bestrho
        #print targetset
    ## For each new target:
    f, x = plt.subplots(1, 1)
    trackrho = [bestrho]
    for i in range(2, ntargetmax):
        st = time()
        ### Optimize new target
        newtarget, newrho = _trynewtarget_construct(cmat, targetset)
        targetset = targetset + [newtarget]
        ### Start MC with candidate:
        bestrho = newrho
        besttset = copy.deepcopy(targetset)
        #### Follow cmat-maximizing trajectory
        ntarget = len(besttset)
        targetset, bestrho, besttset = _stepping_gammamax(targetset, cmat,
                bestrho, besttset, gammamaxmap, allchoices,
                nstep_gammamax, pstep_gammamax, kT, minstep=1)
        #### Random trials
        targetset, bestrho, besttset = _stepping_random(targetset,
                cmat, bestrho, besttset, allchoices,
                nstep_random, pstep_random, kT, minstep=1)
        ### Record rhodata, targetsetdata
        targetset = besttset
        key = (datalabel, ntarget)
        rhodata[key] = bestrho
        tsetdata[key] = targetset
        trackrho.append(bestrho)
        x.cla()
        x.plot(np.arange(2, len(targetset) + 1), trackrho)
        x.set_title(plttitle)
        f.canvas.draw()
        en = time()
        print i + 1, ('(%.2e secs):' % (en - st)), bestrho
        #print targetset
    plt.close(f)
    return rhodata, tsetdata


def _ConstructTset_optimize2(cmat, rhofunc=_rhoindex):
    """
    Optimize 2-target set by exhaustive combinatorial search.
    """
    nbins = len(cmat)
    st = time()
    targetset = [0, 1]
    bestrho = rhofunc(cmat, targetset)
    for i in range(nbins):
        for j in range(i + 1, nbins):
            trialset = [i, j]
            thisrho = rhofunc(cmat, trialset)
            if thisrho < bestrho:
                bestrho = thisrho
                targetset = trialset
    en = time()
    print 2, ('(%.2e secs):' % (en - st)), bestrho
    return bestrho, targetset


def _ConstructTset_dictpivec2(pars, rhomode='frac'):
    """
    Select 2-target set by taking from data dict, or initializing from
    stationary probabilities pi.
    """
    res = pars['res']
    resname = str(res / 1000) + 'kb'
    beta = pars['beta']
    dataset = _get_tsetdataset2(pars)
    chrfullname = pars['chrfullname']
    region = pars['region']
    tsetdatadir = pars['tsetdatadir']
    tsetdataprefix = pars['tsetdataprefix']
    basethreshold = pars['threshold']
    norm = pars['norm']
    rhomodesfx = _get_rhomodesfx(rhomode)
    rhodict0, tsetdict0 = _get_rhotsetdicts_20160801(tsetdatadir,
            tsetdataprefix, chrfullname, region, res, dataset,
            rhomodesfx=rhomodesfx)
    key = (beta, 2)
    if key in rhodict0:
        if key in tsetdict0:
            bestrho = rhodict0[key]
            targetset = tsetdict0[key]
        else:
            print 'Rhodict error: Erase entry', key
            del(rhodict0[key])
            dirname = os.path.join(tsetdatadir, tsetdataprefix,
                        chrfullname, region, resname)
            rfname = os.path.join(dirname, dataset + '-rhodict' +
                            rhomodesfx + '.p')
            _pickle_securedump(rfname, rhodict0, freed=True)
    if key in rhodict0:
        print 'Load 2-target state from data'
        bestrho = rhodict0[key]
        targetset = tsetdict0[key]
    else:
        print 'Seed 2-target state from pivec'
        fmat, _, cmat, _ = _get_arrays(_get_runbinarydir(pars),
                        norm=pars['norm'])
        pivec = np.sum(fmat, axis=1)
        targetset = [np.argmax(pivec)]
        newtarget, bestrho = _trynewtarget_construct(cmat, targetset)
        targetset = targetset + [newtarget]
    del rhodict0, tsetdict0
    return bestrho, targetset


def _ConstructTset_dictpivec2_indict(beta, arrays, indicts):
    """
    Select 2-target set by taking from data dict, or initializing from
    stationary probabilities pi.
    """
    fmat, cmat = arrays
    rhodict0, tsetdict0 = indicts
    key = (beta, 2)
    if key in rhodict0:
        if key in tsetdict0:
            bestrho = rhodict0[key]
            targetset = tsetdict0[key]
        else:
            print 'Rhodict error: Erase entry', key
            del(rhodict0[key])
    if key in rhodict0:
        print 'Load 2-target state from data'
        bestrho = rhodict0[key]
        targetset = tsetdict0[key]
    else:
        print 'Seed 2-target state from pivec'
        pivec = np.sum(fmat, axis=1)
        targetset = [np.argmax(pivec)]
        newtarget, bestrho = _trynewtarget_construct(cmat, targetset)
        targetset = targetset + [newtarget]
    del rhodict0, tsetdict0
    return bestrho, targetset


def _ConstructTset_MCn(steppars, newrho, targetset, allchoices, cmat,
                gammamaxmap, rhofunc=_rhoindex):
    """
    MC steps, starting from preliminary target set.
    """
    #print targetset
    nstep_gammamax, pstep_gammamax, nstep_random, pstep_random, kT = steppars
    ### Start MC with candidate:
    bestrho = newrho
    besttset = copy.deepcopy(targetset)
    #### Follow cmat-maximizing trajectory
    targetset, bestrho, besttset = _stepping_gammamax(targetset, cmat,
            bestrho, besttset, gammamaxmap, allchoices,
            nstep_gammamax, pstep_gammamax, kT, minstep=1, rhofunc=rhofunc)
    #### Random trials
    targetset, bestrho, besttset = _stepping_random(targetset,
            cmat, bestrho, besttset, allchoices,
            nstep_random, pstep_random, kT, minstep=1, rhofunc=rhofunc)
    return bestrho, besttset


def run_ConstructMC_fullarray(pars, steppars, meansize=1.333, initmode=False,
                exhaustive=False, rhomode='frac'):
    """
    Run Construct-MC optimization for rho on a range of ntarget: [2, ntargetmax]
    Set ntargetmax / mappedlength ~ 0.75 Mb^-1
    """
    res = pars['res']
    beta = pars['beta']
    dataset = _get_tsetdataset2(pars)
    chrfullname = pars['chrfullname']
    region = pars['region']
    tsetdatadir = pars['tsetdatadir']
    tsetdataprefix = pars['tsetdataprefix']
    basethreshold = pars['threshold']
    norm = pars['norm']
    nstep_gammamax, pstep_gammamax, nstep_random, pstep_random, kT = steppars
    rhodata = {}
    tsetdata = {}
    resname = str(res / 1000) + 'kb'
    # Rho mode
    rhofunc = _get_rhofunc(rhomode)
    # Read cmat, fmat, mmat, mapping
    fmat, mmat, cmat, (mapping, _) = _get_arrays(
                        _get_runbinarydir(pars), norm=norm)
    nbins = len(cmat)
    allchoices = range(nbins)
    plttitle = '$\\beta=%.1lf$, %s' % (beta, resname)
    # Find gamma-maximizing mapping
    gammamaxmap = np.array([np.argmax(v) for v in cmat])
    # Find ntargetmax
    if initmode:
        ntargetmax = 3
    else:
        mappedlen = (nbins * res) / 1.0e6
        ntargetmax = int(np.ceil(1.0 / meansize * mappedlen))
    if len(cmat) <= ntargetmax:
        print 'Skip...'
        return {}, {}
    print 'ntargetmax =', ntargetmax
    print
    # Find seed target pair: Optimal from current data, or pivec-seeded
    if initmode and exhaustive:
        bestrho, targetset = _ConstructTset_optimize2(cmat, rhofunc=rhofunc)
        if len(targetset) != 2:
            print 'Error! ntarget mismatch 0.0!'
            sys.exit()
        key = (beta, 2)
        rhodata[key] = bestrho
        tsetdata[key] = copy.deepcopy(targetset)
    else:
        st = time()
        newrho, targetset = _ConstructTset_dictpivec2(pars, rhomode=rhomode)
        if len(targetset) != 2:
            print 'Error! ntarget mismatch 0.2!'
            sys.exit()
        bestrho, besttset = _ConstructTset_MCn(steppars, newrho,
                        targetset, allchoices, cmat, gammamaxmap,
                        rhofunc=rhofunc)
        targetset = list(besttset)
        if len(targetset) != 2:
            print 'Error! ntarget mismatch 0.5!'
            sys.exit()
        key = (beta, 2)
        rhodata[key] = bestrho
        tsetdata[key] = copy.deepcopy(targetset)
        en = time()
        print 2, ('(%.2e secs):' % (en - st)), bestrho
    ## For each new target:
    #print type(targetset), targetset
    for i in range(2, ntargetmax):
        ntarget = i + 1
        st = time()
        ### Optimize new target
        #print targetset, 'a'
        newtarget, newrho = _trynewtarget_construct(cmat, targetset,
                        rhofunc=rhofunc)
        #print targetset, 'b'
        targetset.append(newtarget)
        if len(targetset) != ntarget:
            print 'Error! ntarget mismatch 1!'
            sys.exit()
        #print targetset, 'c'
        bestrho, besttset = _ConstructTset_MCn(steppars, newrho,
                        targetset, allchoices, cmat, gammamaxmap,
                        rhofunc=rhofunc)
        ### Record rhodata, targetsetdata
        targetset = besttset
        if len(targetset) != ntarget:
            print 'Error! ntarget mismatch 2!'
            sys.exit()
        key = (beta, ntarget)
        rhodata[key] = bestrho
        tsetdata[key] = copy.deepcopy(targetset)
        en = time()
        print i + 1, ('(%.2e secs):' % (en - st)), bestrho
        #print targetset
    return rhodata, tsetdata


def run_ConstructMC_fullarray_trial(arrays, indicts, pars, steppars,
            meansize=1.333, initmode=False, exhaustive=False, rhomode='frac'):
    """
    Run Construct-MC optimization for rho on a range of ntarget: [2, ntargetmax]
    Set ntargetmax / mappedlength ~ 0.75 Mb^-1
    """
    fmat, cmat = arrays
    res = pars['res']
    beta = pars['beta']
    nstep_gammamax, pstep_gammamax, nstep_random, pstep_random, kT = steppars
    inrdict, intdict = indicts
    rhodata = {}
    tsetdata = {}
    resname = str(res / 1000) + 'kb'
    # Rho mode
    rhofunc = _get_rhofunc(rhomode)
    # Read cmat, fmat, mmat, mapping
    nbins = len(cmat)
    allchoices = range(nbins)
    plttitle = '$\\beta=%.1lf$, %s' % (beta, resname)
    # Find gamma-maximizing mapping
    gammamaxmap = np.array([np.argmax(v) for v in cmat])
    # Find ntargetmax
    if initmode:
        ntargetmax = 2
    else:
        mappedlen = (nbins * res) / 1.0e6
        ntargetmax = int(np.ceil(1.0 / meansize * mappedlen))
    print 'ntargetmax =', ntargetmax
    print
    # Find seed target pair: Optimal from current data, or pivec-seeded
    if initmode and exhaustive:
        bestrho, targetset = _ConstructTset_optimize2(cmat, rhofunc=rhofunc)
        if len(targetset) != 2:
            print 'Error! ntarget mismatch 0.0!'
            sys.exit()
        key = (beta, 2)
        rhodata[key] = bestrho
        tsetdata[key] = copy.deepcopy(targetset)
    else:
        st = time()
        newrho, targetset = _ConstructTset_dictpivec2_indict(
                            beta, arrays, indicts)
        if len(targetset) != 2:
            print 'Error! ntarget mismatch 0.2!'
            sys.exit()
        bestrho, besttset = _ConstructTset_MCn(steppars, newrho,
                        targetset, allchoices, cmat, gammamaxmap,
                        rhofunc=rhofunc)
        targetset = list(besttset)
        if len(targetset) != 2:
            print 'Error! ntarget mismatch 0.5!'
            sys.exit()
        key = (beta, 2)
        rhodata[key] = bestrho
        tsetdata[key] = copy.deepcopy(targetset)
        en = time()
        print 2, ('(%.2e secs):' % (en - st)), bestrho
    ## For each new target:
    #print type(targetset), targetset
    for i in range(2, ntargetmax):
        ntarget = i + 1
        st = time()
        ### Optimize new target
        #print targetset, 'a'
        newtarget, newrho = _trynewtarget_construct(cmat, targetset,
                        rhofunc=rhofunc)
        #print targetset, 'b'
        targetset.append(newtarget)
        if len(targetset) != ntarget:
            print 'Error! ntarget mismatch 1!'
            sys.exit()
        #print targetset, 'c'
        bestrho, besttset = _ConstructTset_MCn(steppars, newrho,
                        targetset, allchoices, cmat, gammamaxmap,
                        rhofunc=rhofunc)
        ### Record rhodata, targetsetdata
        targetset = besttset
        if len(targetset) != ntarget:
            print 'Error! ntarget mismatch 2!'
            sys.exit()
        key = (beta, ntarget)
        rhodata[key] = bestrho
        tsetdata[key] = copy.deepcopy(targetset)
        en = time()
        print i + 1, ('(%.2e secs):' % (en - st)), bestrho
        #print targetset
    return rhodata, tsetdata


def run_ConstructMC_fullarray_testing(arrays, indicts, beta, steppars,
            meansize=10, initmode=False, exhaustive=False, rhomode='frac'):
    """
    Run Construct-MC optimization for rho on a range of ntarget: [2, ntargetmax]
    Set ntargetmax with average partition size = 10 (meansize) pixels
    """
    fmat, cmat = arrays
    nstep_gammamax, pstep_gammamax, nstep_random, pstep_random, kT = steppars
    inrdict, intdict = indicts
    rhodata = {}
    tsetdata = {}
    # Rho mode
    rhofunc = _get_rhofunc(rhomode)
    # Read cmat, fmat, mmat, mapping
    nbins = len(cmat)
    allchoices = range(nbins)
    # Find gamma-maximizing mapping
    gammamaxmap = np.array([np.argmax(v) for v in cmat])
    # Find ntargetmax
    if initmode:
        ntargetmax = 2
    else:
        mappedlen = float(nbins)
        ntargetmax = int(np.ceil(1.0 / meansize * mappedlen))
    print 'ntargetmax =', ntargetmax
    print
    # Find seed target pair: Optimal from current data, or pivec-seeded
    if initmode and exhaustive:
        bestrho, targetset = _ConstructTset_optimize2(cmat, rhofunc=rhofunc)
        if len(targetset) != 2:
            print 'Error! ntarget mismatch 0.0!'
            sys.exit()
        key = (beta, 2)
        rhodata[key] = bestrho
        tsetdata[key] = copy.deepcopy(targetset)
    else:
        st = time()
        newrho, targetset = _ConstructTset_dictpivec2_indict(
                            beta, arrays, indicts)
        if len(targetset) != 2:
            print 'Error! ntarget mismatch 0.2!'
            sys.exit()
        bestrho, besttset = _ConstructTset_MCn(steppars, newrho,
                        targetset, allchoices, cmat, gammamaxmap,
                        rhofunc=rhofunc)
        targetset = list(besttset)
        if len(targetset) != 2:
            print 'Error! ntarget mismatch 0.5!'
            sys.exit()
        key = (beta, 2)
        rhodata[key] = bestrho
        tsetdata[key] = copy.deepcopy(targetset)
        en = time()
        print 2, ('(%.2e secs):' % (en - st)), bestrho
    ## For each new target:
    #print type(targetset), targetset
    for i in range(2, ntargetmax):
        ntarget = i + 1
        st = time()
        ### Optimize new target
        #print targetset, 'a'
        newtarget, newrho = _trynewtarget_construct(cmat, targetset,
                        rhofunc=rhofunc)
        #print targetset, 'b'
        targetset.append(newtarget)
        if len(targetset) != ntarget:
            print 'Error! ntarget mismatch 1!'
            sys.exit()
        #print targetset, 'c'
        bestrho, besttset = _ConstructTset_MCn(steppars, newrho,
                        targetset, allchoices, cmat, gammamaxmap,
                        rhofunc=rhofunc)
        ### Record rhodata, targetsetdata
        targetset = besttset
        if len(targetset) != ntarget:
            print 'Error! ntarget mismatch 2!'
            sys.exit()
        key = (beta, ntarget)
        rhodata[key] = bestrho
        tsetdata[key] = copy.deepcopy(targetset)
        en = time()
        print i + 1, ('(%.2e secs):' % (en - st)), bestrho
        #print targetset
    return rhodata, tsetdata


def _combine_rhotsetdicts(r1, t1, r2, t2):
    rhodict = {}
    tsetdict = {}
    for k in r1:
        if k in r2 and r2[k] < r1[k]:
            rhodict[k] = r2[k]
            tsetdict[k] = t2[k]
        else:
            rhodict[k] = r1[k]
            tsetdict[k] = t1[k]
    for k in r2:
        if k not in r1:
            rhodict[k] = r2[k]
            tsetdict[k] = t2[k]
    return rhodict, tsetdict


################################################################

# Consensus target set determination

## Equal size k-means clustering


def _samesizecluster(D):
    """ in: point-to-cluster-centre distances D, Npt x C
            e.g. from scipy.spatial.distance.cdist
        out: xtoc, X -> C, equal-size clusters
        method: sort all D, greedy
    """
        # could take only the nearest few x-to-C distances
        # add constraints to real assignment algorithm ?
    Npt, C = D.shape
    clustersize = (Npt + C - 1) / C
    xcd = list(np.ndenumerate(D))  # ((0,0), d00), ((0,1), d01) ...
    xcd.sort(key=itemgetter(1))
    xtoc = np.ones(Npt, int) * -1
    nincluster = np.zeros(C, int)
    nall = 0
    for (x, c), d in xcd:
        if xtoc[x] < 0 and nincluster[c] < clustersize:
            xtoc[x] = c
            nincluster[c] += 1
            nall += 1
            if nall >= Npt:
                break
    return xtoc


def sameSizeCluster_1d(points, nclusters, errorcutoff=1.0, maxiter=1000):
    """
    Equal-size k-means clustering of real numbers.
    """
    newctrs = np.random.choice(points, size=nclusters, replace=False)
    error = 100.0
    niter = 0
    while error > errorcutoff and niter < maxiter:
        niter += 1
        ctrs = newctrs.copy()
        dists = np.array([[np.abs(c - p) for c in ctrs] for p in points])
        clustering = _samesizecluster(dists)
        newctrs = np.array([np.average(np.array(points)[clustering == i])
                        for i in range(nclusters)])
        error = np.sum(np.abs(ctrs - newctrs))
    return np.array([np.array(points)[clustering == i]
                    for i in range(nclusters)])


def ConsensusTset_kmeans(mappedtargets, ntargets, errorcutoff=1.0):
    """
    Construct consensus target set by k-means clustering of mapped targets.
    Return list of candidate targets
    """
    clustering = sameSizeCluster_1d(mappedtargets, ntargets,
                    errorcutoff=errorcutoff)
    candidates = [-1 for i in range(ntargets)]
    for i, cluster in enumerate(clustering):
        clusteruniq = list(set(cluster))
        clusteruniq.sort()
        clusterpoll = np.array([list(cluster).count(mt) for mt in clusteruniq])
        # Select all target candidates with highest in-group poll
        pollmax = np.max(clusterpoll)
        candidates[i] = tuple(np.array(clusteruniq)[clusterpoll == pollmax])
    sortinds = np.argsort(map(np.average, candidates))
    candidates = map(tuple, np.array(candidates)[sortinds])
    clustering = np.array(clustering)[sortinds]
    return candidates, clustering


def ConsensusTset_kmeans_sort(mappedtargets, ntargets, errorcutoff=1.0):
    """
    Sort k-means clustered targets by occurrence ranking. most frequent first.
    """
    mappedtargetset = np.sort(list(set(mappedtargets)))
    mappedtargetpoll = [mappedtargets.count(t) for t in mappedtargetset]
    candidates, clustering = ConsensusTset_kmeans(mappedtargets, ntargets,
                    errorcutoff=errorcutoff)
    rankedclustering = []
    for thiscluster in clustering:
        thisclusteruniq = np.unique(thiscluster)
        thisranks = [mappedtargetpoll[list(mappedtargetset).index(t)]
                        for t in thisclusteruniq]
        rankedclustering.append(thisclusteruniq[np.argsort(thisranks)[::-1]].copy())
    return candidates, rankedclustering


####################################################

# Raw Hi-C data file reading


def _extract_fij_subregion_LiebermannAiden2014(chrname,
                    genomeref, genomedatadir, hicfile,
                    baseres, baseresname, res, regionselection, nloop=0,
                    weightboundaries=False, norm='raw', minrowsum=0.0):
    """
    Extract interaction matrix for subregion defined by selection.
    """
    # Find data size nbins
    nbins = _get_chrdatasize(chrname, genomeref, genomedatadir, res)
    # Create mapping arrays
    sliceselections, sliceweights, slicetopixels, pixelweights, mapping = \
                    _calc_subsetmappingdata(regionselection, baseres, res)
    # Initialize CG data array
    npx = len(mapping)
    fmat = np.zeros((npx, npx))
    # Map data to fmat
    minpos = np.min(sliceselections)
    maxpos = np.max(sliceselections)
    f = open(hicfile, 'r')
    for line in f:
        ## Increment pixel and the symmetric element
        i, j, fij = line.split()
        pos1 = int(i)
        if pos1 < minpos:
            continue
        pos2 = int(j)
        if pos2 < minpos:
            continue
        elif pos2 > maxpos and pos1 > maxpos:
            break
        val = float(fij)
        if pos1 in sliceselections and pos2 in sliceselections:
            x = sliceselections.index(pos1)
            y = sliceselections.index(pos2)
            fmat[slicetopixels[x], slicetopixels[y]] += \
                            val * sliceweights[x] * sliceweights[y]
            if pos1 != pos2:
                fmat[slicetopixels[y], slicetopixels[x]] += \
                                val * sliceweights[x] * sliceweights[y]
    f.close()
    # Normalization
    if norm == 'raw':
        pass
    elif norm == 'KR' or norm == 'VC' or norm == 'SQRTVC':
        # Read norm vector
        normfile = hicfile[:hicfile.find('RAWobserved')] + norm + 'norm'
        normvec = []
        with open(normfile, 'r') as f:
            for line in f:
                val = float(line)
                # If 0.0 or nan, set to inf
                if val == 0.0 or not np.isfinite(val):
                    val = np.inf
                normvec.append(val)
        normvec = np.array(normvec)
        # Truncate normvec
        normvec = normvec[:len(fmat)]
        # Divide fmat by outer product
        fmat /= np.outer(normvec, normvec)
    else:
        print 'Invalid normalization mode', norm, '!'
        sys.exit(1)
    hasdata = (pixelweights > 0.0) * (np.sum(fmat, 0) > minrowsum)
    if np.sum(hasdata) < len(pixelweights):
        pixelweights = pixelweights[hasdata]
        mapping = np.array(mapping)[hasdata]
        fmat = fmat[hasdata][:, hasdata]
        print
    if weightboundaries:
        fmat /= np.outer(pixelweights, pixelweights)
    fmat += (nloop - 1.0) * np.diag(np.diag(fmat))
    # Return
    return fmat, (mapping, nbins)

