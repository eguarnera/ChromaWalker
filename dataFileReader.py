#!/usr/bin/env python
"""
Collection of data-file handling functions and utilities for Hi-C analysis

Part of ChromaWalker package
"""


import numpy as np
import os
import sys
import copy
from time import sleep
import cPickle as pickle
import hicFileIO as hfio
import msmBasics as mb
import msmTPT as mt
import matplotlib.pyplot as plt
import plotutils as plu
from scipy.ndimage.filters import gaussian_filter


#################################################

## Reading data files: ASCII files


def _get_allchrsizes(chrnames, genomeref, genomedatadir):
    """
    Find out how many pixels are required to represent
    given chromosome at given resolution.
    Reads cached UCSC data file *.chrom.sizes
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
    Read cytoband data for given chromosome and create array
      representation at given resolution.
    """
    genomedatadir = pars['genomedatadir']
    genomeref = pars['genomeref']
    chrname = pars['cname']
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


## Extract Hi-C data from ASCII data file


def _calc_subsetmappingdata(selection, baseres, res):
    """
    Calculate data required to extract subregion interaction matrix
    from sparse-format file.
    See docstring for _extract_fij_subregion_LiebermannAiden2014().
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
    elif norm.startswith('gfilter'):
        # Extract filter rad, sigma
        sigma = float(norm.split('_')[1])
        sigma /= res
        # Get filtered fmat
        masksm = gaussian_filter((fmat > 0.0) * 1.0, sigma, mode='constant')
        masksm[masksm == 0.0] = np.inf
        fmat = gaussian_filter(fmat, sigma, mode='constant') / masksm
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


def _extract_fij_subregion_LiebermannAiden2014_inter(chrname1, chrname2,
                    genomeref, genomedatadir, hicfile,
                    baseres, baseresname, res, regionselection1,
                    regionselection2, weightboundaries=False,
                    norm='raw'):
    """
    Extract interaction matrix for subregion defined by selection.
    """
    # Find data size nbins
    nbins1 = _get_chrdatasize(chrname1, genomeref, genomedatadir, res)
    nbins2 = _get_chrdatasize(chrname2, genomeref, genomedatadir, res)
    # Create mapping arrays
    sliceselections1, sliceweights1, slicetopixels1, pixelweights1, mapping1 = \
                    _calc_subsetmappingdata(regionselection1, baseres, res)
    sliceselections2, sliceweights2, slicetopixels2, pixelweights2, mapping2 = \
                    _calc_subsetmappingdata(regionselection2, baseres, res)
    # Initialize CG data array
    npx1 = len(mapping1)
    npx2 = len(mapping2)
    fmat = np.zeros((npx1, npx2))
    # Map data to fmat
    minpos1 = np.min(sliceselections1)
    maxpos1 = np.max(sliceselections1)
    minpos2 = np.min(sliceselections2)
    maxpos2 = np.max(sliceselections2)
    f = open(hicfile, 'r')
    for line in f:
        ## Increment pixel and the symmetric element
        i, j, fij = line.split()
        pos1 = int(i)
        if pos1 < minpos1:
            continue
        pos2 = int(j)
        if pos2 < minpos2:
            continue
        elif pos2 > maxpos2 and pos1 > maxpos1:
            break
        val = float(fij)
        if pos1 in sliceselections1 and pos2 in sliceselections2:
            x = sliceselections1.index(pos1)
            y = sliceselections2.index(pos2)
            fmat[slicetopixels1[x], slicetopixels2[y]] += \
                            val * sliceweights1[x] * sliceweights2[y]
    f.close()
    # Normalization
    if norm == 'raw':
        pass
    elif norm == 'KR' or norm == 'VC' or norm == 'SQRTVC':
        # Get data directory
        datadir = os.path.dirname(hicfile)
        # Find both norm vectors
        tempresname = str(res / 1000) + 'kb'
        chrfullname1 = chrname1 if chrname1.find('chr') == 0 else \
                        'chr' + chrname1
        chrfullname2 = chrname2 if chrname2.find('chr') == 0 else \
                        'chr' + chrname2
        normfile1 = os.path.join(datadir, chrfullname1 + '_'
                        + tempresname + '.' + norm + 'norm')
        normfile2 = os.path.join(datadir, chrfullname2 + '_'
                        + tempresname + '.' + norm + 'norm')
        normvec1 = []
        normvec2 = []
        with open(normfile1, 'r') as f:
            for line in f:
                val = float(line)
                # If 0.0 or nan, set to inf
                if val == 0.0 or not np.isfinite(val):
                    val = np.inf
                normvec1.append(val)
        with open(normfile2, 'r') as f:
            for line in f:
                val = float(line)
                # If 0.0 or nan, set to inf
                if val == 0.0 or not np.isfinite(val):
                    val = np.inf
                normvec2.append(val)
        normvec1 = np.array(normvec1)
        normvec2 = np.array(normvec2)
        # Truncate and divide
        normvec1 = normvec1[:fmat.shape[0]]
        normvec2 = normvec2[:fmat.shape[1]]
        fmat /= np.outer(normvec1, normvec2)
    else:
        print 'Invalid normalization mode', norm, '!'
        sys.exit(1)
    hasdata1 = (pixelweights1 > 0.0) * (np.sum(fmat, 1) > 0.0)
    hasdata2 = (pixelweights2 > 0.0) * (np.sum(fmat, 0) > 0.0)
    if np.sum(hasdata1) < len(pixelweights1) or \
                    np.sum(hasdata2) < len(pixelweights2):
        pixelweights1 = pixelweights1[hasdata1]
        pixelweights2 = pixelweights2[hasdata2]
        mapping1 = np.array(mapping1)[hasdata1]
        mapping2 = np.array(mapping2)[hasdata2]
        fmat = fmat[hasdata1][:, hasdata2]
        print
    if weightboundaries:
        fmat /= np.outer(pixelweights1, pixelweights2)
    # Return
    return fmat, (mapping1, nbins1), (mapping2, nbins2)


################################################

# Retrieving binary archives


def _get_runbinarydir(pars):
    """
    Get directory pointing to binary data archive.
    For intra-chromosomal data.
    """
    rundir = pars['rundir']
    chrref = pars['cname']
    region = pars['region']
    accession = pars['accession']
    runlabel = pars['runlabel']
    beta = pars['beta']
    res = pars['res']
    resname = str(res / 1000) + 'kb'
    minrowsum = pars['minrowsum']
    nloop = pars['nloop']
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
    For inter-chromosomal data.
    """
    rundir = pars['rundir']
    accession = pars['accession']
    runlabel = pars['runlabel']
    chrfullname1 = pars['cname1']
    chrfullname2 = pars['cname2']
    region = pars.get('region', 'full')
    res = pars['res']
    beta = pars.get('beta', 1.0)
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
        mmat = mb._calc_MFPT(fmat)
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


def _get_fmatmap_inter(pars):
    """
    Get inter-chromosomal interaction fmat, from binary file, or from
    sparse ASCII file.
    Also gives mapping data.
    Full chromosome selections only.
    """
    # Parse parameters
    norm = pars.get('norm', 'raw')
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
        return fmat, md1, md2
    else:
        return


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
    tsetmap = hfio._pickle_secureread(fname, free=False)
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
        if np.allclose(np.sort(targetset), np.sort(tset)) and n == norm:
            tsetindex = index
            tsetorder = np.argsort(targetset)
            break
    if tsetindex is None:
        print 'Create new entry in tsetmap...'
        #print fname
        targetset = copy.deepcopy(tset)
        # Update dict
        tsetindex = len(tsetmap)
        tsetmap[tsetindex] = tset, norm
        hfio._pickle_securedump(fname, tsetmap, freed=freed)
    else:
        # Manually remove lock
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
        qAi = mt._calc_qAi_sort_exact(datadir, targetset, norm=norm)
        qAi.tofile(fname_TC)
    return qAi


################################################

# Pickled files: Actually, just the dicts for rho and target sets


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
        rdict, tdict = hfio._pickle_securereads([outrfname, outtfname],
                        free=free)
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


def _update_datadicts(rdata, tsetdata, tsetdatadir, tsetdataprefix,
                chrfullname, region, res, dataset, rhomode='frac'):
    """
    Update database with output from run_ConstructMC_fullarray
    Version 20160801: Split data dicts
    Returns number of updated entries.
    """
    nupdate = 0
    rhomodesfx = mt._get_rhomodesfx(rhomode)
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
            nupdate += 1
    # Write to file
    resname = str(res / 1000) + 'kb'
    dirname = os.path.join(tsetdatadir, tsetdataprefix,
                    chrfullname, region, resname)
    rfname = os.path.join(dirname, dataset + '-rhodict' + rhomodesfx + '.p')
    tfname = os.path.join(dirname, dataset + '-tsetdict' + rhomodesfx + '.p')
    hfio._pickle_securedumps((rfname, tfname), (rdict, tdict), freed=False)
    return nupdate


################################################

# To deprecate...


def _get_runbinarydir_old(pars):
    """
    Get directory pointing to binary data archive.
    For intra-chromosomal data.
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


#################################################################

# ReadDataFile class:
#    - Reading / dumping fmat, mmat, cmat
#    - Reading, updating tset / rho dicts
#    - Reading / dumping committor arrays


class DataFileReader:
    """
    Container to hold parameters pertaining to HiC analysis runs and
    simplify function calls, for data retrieval / dumping.
    In some instances, a wrapper to interface with legacy code that
        I'm too lazy to modify.
    """
    def __init__(self, pars, epigenpars=None):
        """
        Initialize ReadDataFile instance
        Input: dicts pars and epigenpars
        Note that pars must contain the following keys:
            'rawdatadir': Location for HiC ASCII data
            'baseres': Resolution of HiC ASCII data
            'genomedatadir': Location for sequence / cytological-level data
            'genomeref': Reference genome
            'rundir': Location for dumping binary data
            'accession': Accession number for input data
            'runlabel': Further identifiers (cell line, replicate, etc.)
            'tsetdatadir': Location for dumping tset/rho data
            'tsetdataprefix': Label for optimization routine used
        Note that epigenpars must contain the following keys:
            'epigendatadir': Location for epigenetic track data (bigWigs)
        """
        # Basic file directory / name info
        self.rawdatadir = pars['rawdatadir']
        self.genomedatadir = pars['genomedatadir']
        self.genomeref = pars['genomeref']
        self.rundir = pars['rundir']
        self.dataformat = 'Liebermann-Aiden'
        self.accession = pars['accession']
        self.runlabel = pars['runlabel']
        self.tsetdatadir = pars['tsetdatadir']
        self.tsetdataprefix = pars['tsetdataprefix']

        #############################
        # Info about dataset / run:
        ## Which genomic regions?
        if 'cnamelist' in pars:
            self.cnamelist = pars['cnamelist']
            self.cname = self.cnamelist[0]
        else:
            self.cnamelist = ['chr' + str(i) for i in range(1, 23)] + ['chrX']
            self.cname = 'chr1'
        self.region = pars.get('region', 'full')
        ## Data resolution?
        self.baseres = pars['baseres']
        self.res = pars.get('res', self.baseres)
        ## Include self-interactions (loops)? How many times?
        self.nloop = pars.get('nloop', 0)
        ## Noise-filtering by thermal annealing
        if 'betalist' in pars:
            self.betalist = pars['betalist']
        else:
            self.betalist = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        self.beta = pars.get('beta', self.betalist[0])
        ## Data normalization by row-normalizing vectors or gaussian filter
        self.norm = pars.get('norm', 'raw')
        ## Noise filtering by filtering out low-count pixels
        ###    To deprecate
        self.basethreshold = pars.get('threshold', 0.0)
        ## Noise filtering by filtering out low-count columns
        ###    To deprecate
        self.minrowsum = pars.get('minrowsum', 0.0)
        ## TPT optimizer function: Whoch rho to use?
        ###    To deprecate
        self.rhomode = pars.get('rhomode', 'frac')

        #############################
        # Info about epigenetic data tracks
        if epigenpars is None or 'epigendatadir' not in epigenpars:
            return
        self.epigendatadir = epigenpars['epigendatadir']
        self.epigentypes = epigenpars.get('epigentypes',
                ['Histone', 'TFBS', 'DNase', 'RNA', 'Methylation'])
        self.eptype = self.epigentypes[0]
        self.tracktypes = epigenpars.get('tracktypes', ['signal', 'coverage'])
        self.trtype = self.tracktypes[0]
        self.binsize = epigenpars.get('binsize', self.res)
        ## Need user to further define source and signal names for retrieval
        ## Future: Automatically load source/signal list from metafile

        #############################
        # Create base pardicts
        self.basepars = {'rawdatadir': self.rawdatadir,
                'rawdatadir': self.rawdatadir,
                'genomedatadir': self.genomedatadir,
                'genomeref': self.genomeref,
                'rundir': self.rundir,
                'dataformat': self.dataformat,
                'accession': self.accession,
                'runlabel': self.runlabel,
                'tsetdatadir': self.tsetdatadir,
                'tsetdataprefix': self.tsetdataprefix,
                'cname': self.cname,
                'region': self.region,
                'baseres': self.baseres,
                'res': self.res,
                'nloop': self.nloop,
                'beta': self.beta,
                'tsetbeta': self.beta,
                'norm': self.norm,
                'threshold': self.basethreshold,
                'minrowsum': self.minrowsum,
                'rhomode': self.rhomode
                }

        ####################################
        self.baseepigenpars = {'epigendatadir': self.epigendatadir,
                'epigentypes': self.epigentypes,
                'eptype': self.eptype,
                'tracktypes': self.tracktypes,
                'trtype': self.trtype,
                'binsize': self.binsize}

    def _get_regionselection(self, cname):
        """
        Internal function: Used as input for functions reading ASCII fmat.
        """
        chrlen = _get_chrdatasize(cname, self.genomeref,
                        self.genomedatadir, self.baseres)
        return [[0, chrlen * self.baseres]]

    def _set_dummyarrays(self, cname, beta):
        """
        Set array files in directory to tiny dummy values. Saves disk space
        when we want to ignore high beta values that don't meet the
        selection criterion.
        """
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname
        thispar['beta'] = beta
        binary_dir = _get_runbinarydir(thispar)
        fmatfname = os.path.join(binary_dir, 'fmat-' + self.norm + '.dat')
        mmatfname = os.path.join(binary_dir, 'mmat-' + self.norm + '.dat')
        cmatfname = os.path.join(binary_dir, 'cmat-' + self.norm + '.dat')
        mapfname = os.path.join(binary_dir, 'mapping-' + self.norm + '.dat')
        np.array([0.0]).tofile(fmatfname)
        np.array([0.0]).tofile(mmatfname)
        np.array([0.0]).tofile(cmatfname)
        hfio._pickle_securedump(mapfname, (np.array([0]), 1), freed=True)

    def get_fmat(self, cname, beta):
        """
        Retrieve single-chr interaction matrix fmat, either by reading from
        binary cache (if available) or from ASCII input file.
        Input: chromosome name, annealing beta
        """
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname
        thispar['beta'] = beta
        binary_dir = _get_runbinarydir(thispar)
        thispar['cname'] = cname
        thispar['beta'] = 1.0
        binary_dir_beta1 = _get_runbinarydir(thispar)
        if not os.path.isdir(binary_dir):
            os.makedirs(binary_dir)
        if not os.path.isdir(binary_dir_beta1):
            os.makedirs(binary_dir_beta1)
        # Test if binary file exists
        fmatfname = os.path.join(binary_dir, 'fmat-' + self.norm + '.dat')
        fmatbeta1fname = os.path.join(binary_dir_beta1, 'fmat-' +
                        self.norm + '.dat')
        if os.path.isfile(fmatfname):
            ## If exist, read binary file
            fmat = np.fromfile(fmatfname, 'float64')
            n = int(np.sqrt(len(fmat)))
            fmat.shape = n, n
        elif os.path.isfile(fmatbeta1fname):
            ## If beta=1.0 data exists, read binary file then exponentiate
            fmat = np.fromfile(fmatbeta1fname, 'float64')
            n = int(np.sqrt(len(fmat)))
            fmat.shape = n, n
            fmat = fmat ** beta
            fmat.tofile(fmatfname)
            md = self.get_mappingdata(cname, 1.0)
            mapfname = os.path.join(binary_dir,
                        'mapping-' + self.norm + '.dat')
            hfio._pickle_securedump(mapfname, md, freed=True)
        else:
            ## Else, read ASCII file and dump binary
            baseresname = str(int(self.baseres / 1000)) + 'kb'
            hicfile = os.path.join(self.rawdatadir, self.accession,
                        self.runlabel,
                        baseresname + '_resolution_intrachromosomal',
                        cname, 'MAPQGE30', cname + '_' +
                        baseresname + '.RAWobserved')
            regionselection = self._get_regionselection(cname)
            fmat, mappingdata = _extract_fij_subregion_LiebermannAiden2014(cname,
                    self.genomeref, self.genomedatadir, hicfile,
                    self.baseres, baseresname, self.res, regionselection,
                    nloop=self.nloop, norm=self.norm, minrowsum=self.minrowsum)
            # Dump beta = 1.0 file first
            fmat.tofile(fmatbeta1fname)
            mapfname = os.path.join(binary_dir_beta1,
                        'mapping-' + self.norm + '.dat')
            hfio._pickle_securedump(mapfname, mappingdata,
                        freed=True)
            # If beta is not 1.0, dump exponentiated version
            if beta != 1.0:
                fmat = fmat ** beta
                fmat.tofile(fmatfname)
                mapfname = os.path.join(binary_dir,
                            'mapping-' + self.norm + '.dat')
                hfio._pickle_securedump(mapfname, mappingdata,
                            freed=True)
        return fmat

    def get_mappingdata(self, cname, beta):
        """
        Retrieve mapping information: mapping, nbins
        Pixel i in f/m/cmat arrays point to bin mapping[i] along genome.
        The whole genome is split into nbins bins.
        """
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname
        thispar['beta'] = beta
        binary_dir = _get_runbinarydir(thispar)
        mapfname = os.path.join(binary_dir,
                    'mapping-' + self.norm + '.dat')
        if not os.path.isfile(mapfname):
            self.get_fmat(cname, beta)
        return hfio._pickle_secureread(mapfname, free=True)


    def get_mmat(self, cname, beta):
        """
        Retrieve MFPT matrix, either by reading from
        binary cache (if available) or computing directly from fmat.

        Note that at high beta, some nodes may become disconnected. We choose
        to mask these nodes to preserve one connected component (which
        hopefully is large).
        If some nodes are masked during this process,
        both the interaction matrix fmat and mapping data are updated.
        """
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname
        thispar['beta'] = beta
        binary_dir = _get_runbinarydir(thispar)
        mmatfname = os.path.join(binary_dir, 'mmat-' + self.norm + '.dat')
        # Test if binary mmat file exists
        if os.path.isfile(mmatfname):
            ## If exist, read binary file
            mmat = np.fromfile(mmatfname, 'float64')
            n = int(np.sqrt(len(mmat)))
            mmat.shape = n, n
        else:
            ## Else, read fmat file and dump binary
            ### Note: May need to trim out rows at high beta due to
            ###       beta-induced disconnected pieces.
            fmat = self.get_fmat(cname, beta)
            mappingdata = self.get_mappingdata(cname, beta)
            fmat2, mmat, mapping = mb._calc_MFPT_20160831(fmat, mappingdata[0])
            if len(fmat2) < len(fmat):
                #### Rows were trimmed out: update fmat, mappingdata
                fmatfname = os.path.join(binary_dir, 'fmat-' +
                                self.norm + '.dat')
                mapfname = os.path.join(binary_dir,
                            'mapping-' + self.norm + '.dat')
                fmat2.tofile(fmatfname)
                hfio._pickle_securedump(mapfname, (mapping, mappingdata[1]),
                            freed=True)
            mmat.tofile(mmatfname)
        return mmat

    def get_cmat(self, cname, beta):
        """
        Retrieve hitting probability matrix, either by reading from
        binary cache (if available) or computing directly from mmat.
        """
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname
        thispar['beta'] = beta
        binary_dir = _get_runbinarydir(thispar)
        cmatfname = os.path.join(binary_dir, 'cmat-' + self.norm + '.dat')
        # Test if binary cmat file exists
        if os.path.isfile(cmatfname):
            ## If exist, read binary file
            cmat = np.fromfile(cmatfname, 'float64')
            n = int(np.sqrt(len(cmat)))
            cmat.shape = n, n
        else:
            ## Else, read mmat file and dump binary
            mmat = self.get_mmat(cname, beta)
            cmat = mb._calc_cmat(mmat)
            cmat.tofile(cmatfname)
        return cmat

    def get_bands(self, cname):
        """
        Retrieve cytoband data.
        """
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname
        return _get_cytoband(thispar)

    def get_arrays(self, cname, beta):
        """
        Get fmat, mmat, cmat, mappingdata.
        """
        mmat = self.get_mmat(cname, beta)
        fmat = self.get_fmat(cname, beta)
        mappingdata = self.get_mappingdata(cname, beta)
        cmat = self.get_cmat(cname, beta)
        return fmat, mmat, cmat, mappingdata

    def get_fmatMapdata_inter(self, cname1, cname2):
        """
        Get inter-chromosomal interaction matrix between chromosomes
        cname1 and cname2.

        We did not implement any form of normalization (row-sum or gaussian
        filter) and thermal annealing (beta > 1.0) to inter-chromosomal data.
        This could be something worth doing in the future when we can
        calculate MFPT on the whole genome, but that seems too far away now...
        """
        thispar = copy.deepcopy(self.basepars)
        thispar['cname1'] = cname1
        thispar['cname2'] = cname2
        thispar['beta'] = 1.0
        thispar['norm'] = 'raw'
        binary_dir = _get_runbinarydir_interchr(thispar)
        # Get inter-chr fmat, as well as the corresponding mappingdata (md)
        # for each chromosome
        # Note that this isn't necessarily the same mappingdata as in
        # single-chromosome data, but simply corresponds to rows with nonzero
        # inter-chr counts in this pair of chromosomes
        if os.path.isdir(binary_dir) and \
                _get_fmatmap_inter(thispar) is not None:
            fmat, md1, md2 = _get_fmatmap_inter(thispar)
        else:
            baseresname = str(int(self.baseres / 1000)) + 'kb'
            # Compute array from sparse ASCII file
            regionselection1 = self._get_regionselection(cname1)
            regionselection2 = self._get_regionselection(cname2)
            hicfile12 = os.path.join(self.rawdatadir, self.accession,
                self.runlabel + '_interchromosomal',
                baseresname + '_resolution_interchromosomal',
                cname1 + '_' + cname2,
                'MAPQGE30', cname1 + '_' + cname2[3:] + '_' +
                        baseresname + '.RAWobserved')
            fmat, md1, md2 = \
                _extract_fij_subregion_LiebermannAiden2014_inter(
                    cname1, cname2, self.genomeref, self.genomedatadir,
                    hicfile12, self.baseres, baseresname, self.res,
                    regionselection1, regionselection2)
            #if self.norm.startswith('gfilter'):
                ## Extract filter rad, sigma
                #sigma = float(self.norm.split('_')[1])
                #sigma /= res
                ## Get filtered fmat
                #masksm = gaussian_filter((fmat > 0.0) * 1.0, sigma, mode='constant')
                #fmat = gaussian_filter(fmat, sigma, mode='constant') / masksm
            binaryfname = os.path.join(binary_dir, 'fmat-' +
                            thispar['norm'] + '.dat')
            mappingdatafname = os.path.join(binary_dir,
                            'mapping-' + thispar['norm'] + '.dat')
            fmat.tofile(binaryfname)
            pickle.dump((md1, md2), open(mappingdatafname, 'wb'))
        return fmat, md1, md2

    def get_datadicts(self, cname):
        """
        Get dictionaries of optimal target sets and corresponding values of rho
        """
        thispar = copy.deepcopy(self.basepars)
        dataset = _get_tsetdataset2(thispar)
        rhomodesfx = mt._get_rhomodesfx(self.rhomode)
        rdict, tdict = _get_rhotsetdicts_20160801(
                self.tsetdatadir, self.tsetdataprefix, cname, self.region,
                self.res, dataset, free=True, rhomodesfx=rhomodesfx)
        return rdict, tdict

    def update_datadicts(self, cname, inrho, intset):
        """
        Update dictionaries of optimal target sets and corresponding
        values of rho, with input dictionaries from an optimization run.
        """
        thispar = copy.deepcopy(self.basepars)
        dataset = _get_tsetdataset2(thispar)
        return _update_datadicts(inrho, intset,
                self.tsetdatadir, self.tsetdataprefix, cname,
                self.region, self.res, dataset, rhomode=self.rhomode)

    def readout_datadicts(self, rdict, tdict, beta, ntarget):
        """
        Read out rho values and target set for given beta and ntarget
        (Just a wrapper so we don't have to remember what the dict keys are)
        Returns None if no match for (beta, ntarget)
        """
        if (beta, ntarget) not in rdict or (beta, ntarget) not in tdict:
            return None
        else:
            rho = rdict[beta, ntarget]
            tset = tdict[beta, ntarget]
            return rho, tset

    def get_committor(self, cname, beta, tset):
        """
        Retrieve committor array
        Note that the current implementation caches binary qAi in the same
        directory as fmat, etc. An indexing dictionary tsetmap is used to
        map to (tset, norm), from an index associated with the file.
        """
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname
        thispar['beta'] = beta
        binary_dir = _get_runbinarydir(thispar)
        qAi = _get_TargetCommittor(binary_dir, tset, norm=self.norm)
        return qAi


if __name__ == '__main__':
    ################################
    # Perform tests on class functions
    plt.ion()
    print
    print '**********************************************'
    print 'Welcome to ReadDataFiles test suite!'
    print '**********************************************'
    print
    baseres = 50000
    res = 50000
    cname = 'chr22'
    cname2 = 'chr21'
    beta = 1.0
    pars = {
            #'rawdatadir': '/home/tanzw/data/hicdata/ProcessedData/',
            #'genomedatadir': '/home/tanzw/data/genomedata/',
            'rawdatadir': 'asciidata/',
            'genomedatadir': 'asciidata/',
            'genomeref': 'hg19',
            'rundir': 'rundata/',
            'accession': 'GSE63525',
            'runlabel': 'GM12878_primary',
            'tsetdatadir': 'rundata/TargetsetOptimization/',
            'tsetdataprefix': 'Full-ConstructMC',
            'baseres': baseres,
            'res': res}
    epigenpars = {'epigendatadir': 'epigenomic-tracks'}
    print 'Create RDF object...'
    dfr = DataFileReader(pars, epigenpars=epigenpars)
    # Get cytoband data
    print 'Getting cytoband data...'
    cytobanddata = dfr.get_bands(cname)
    # Load and compute f/m/cmat
    print 'Compute f/m/cmat...'
    fmat, mmat, cmat, mappingdata = dfr.get_arrays(cname, beta)
    # Trial tset: first and last pixels
    print 'Try tset...'
    ntarget = 2
    lousytset = [0, len(mappingdata[0]) - 1]
    rd, td = dfr.get_datadicts(cname)
    rd1 = {(beta, ntarget): mt._rhoindex(cmat, lousytset)}
    td1 = {(beta, ntarget): lousytset}
    dfr.update_datadicts(cname, rd1, td1)
    rd, td = dfr.get_datadicts(cname)
    rhoval, tsetval = dfr.readout_datadicts(rd, td, beta, ntarget)
    print 'Compute qAi...'
    # Calculate qAi
    qAi = dfr.get_committor(cname, beta, lousytset)
    ## Recalculate: Should just load from file
    qAi2 = dfr.get_committor(cname, beta, lousytset)
    # Get inter-chromosomal interaction matrix
    print 'Getting inter-chromosomal interaction matrix...'
    fmat1 = dfr.get_fmat(cname, 1.0)
    mappingdata1 = dfr.get_mappingdata(cname, 1.0)
    fmat2 = dfr.get_fmat(cname2, 1.0)
    mappingdata2 = dfr.get_mappingdata(cname2, 1.0)
    fmat21, md2, md1 = dfr.get_fmatMapdata_inter(cname2, cname)
    print
    print '**********************************************'
    print
    ################################
    # Display test results
    print 'Chromosome %s, resolution %i' % (cname[3:], res)
    print 'Data array sizes: %i, %i, %i, %i' % (len(fmat), len(mmat),
            len(cmat), len(mappingdata[0]))
    print 'nbins: %i' % mappingdata[1]
    print 'Trial tset [%i, %i]: rho = %e' % (tsetval[0], tsetval[1],
                        rhoval)
    f, x = plt.subplots(1, 1, figsize=(8, 1))
    plu._plot_cytobands(cytobanddata, res, x, plotbname=True)
    f = []
    x = []
    for i in range(3):
        a, b = plt.subplots(1, 1, figsize=(7, 6))
        f.append(a)
        x.append(b)
    _ = plt.colorbar(x[0].matshow(np.log(fmat) / np.log(10.0)), ax=x[0])
    _ = plt.colorbar(x[1].matshow(np.log(mmat) / np.log(10.0)), ax=x[1])
    _ = plt.colorbar(x[2].matshow(cmat), ax=x[2])
    _ = x[0].set_title('$\log_{10}(f_{ij})$')
    _ = x[1].set_title('$\log_{10}(m_{ij})$')
    _ = x[2].set_title('$c_{ij}$')
    _ = raw_input('Cytoband data and f/m/cmat arrays. ' +
                    'Enter anything to continue:')
    plt.close('all')
    print
    f, x = plt.subplots(1, 1, figsize=(8, 3))
    _ = x.plot(qAi.T)
    _ = x.set_title('Trial $q_\\alpha(i)$')
    fmat1pad = plu._build_fullarray(fmat1, mappingdata1, np.nan)
    fmat2pad = plu._build_fullarray(fmat2, mappingdata2, np.nan)
    fmat21pad = plu._build_fullarray_inter(fmat21, md2, md1, np.nan)
    fillmat = np.array(np.bmat([[fmat2pad, fmat21pad], [fmat21pad.T, fmat1pad]]))
    f, x = plt.subplots(1, 1, figsize=(6, 6))
    _ = plt.colorbar(x.matshow(np.log(fillmat) / np.log(10.0)), ax=x)
    _ = x.set_title('%s-%s $\log_{10}(f_{ij})$' % (cname2, cname))
    print
    print
    _ = raw_input('Enter anything to exit:')
    plt.close('all')
    print 'Farewell!'
    print


