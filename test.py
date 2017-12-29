# Collection of analysis functions and utilities
# Specialized for Hi-C analysis

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import random
import copy
from time import time
import hicutils as hcu
import hicFileIO as hfio
import readDataFiles as rdf
import msmTPT as mt
import msmTsetOpt as mto
from operator import itemgetter



##############################################################

# Test readDataFiles

## Reading ASCII files


def test_get_allchrsizes(genomedatadir):
    """
    Find out how many pixels are required to represent
    given chromosome at given resolution.
    Reads cached UCSC data file *.chrom.sizes
    """
    chrnames, genomeref = ['chr' + str(i) for i in range(1, 23)] + ['chrX'], \
                    'hg19'
    datavals = rdf._get_allchrsizes(chrnames, genomeref, genomedatadir)
    refvals = [249250621, 243199373, 198022430, 191154276, 180915260,
                171115067, 159138663, 146364022, 141213431, 135534747,
                135006516, 133851895, 115169878, 107349540, 102531392,
                90354753, 81195210, 78077248, 59128983, 63025520, 48129895,
                51304566, 155270560]
    nbad = 0
    nchrs = len(chrnames)
    print 'Testing _get_allchrsizes()...'
    for ichr, (d, v) in enumerate(zip(datavals, refvals)):
        if d == v:
            pass
            #print 'Chromosome %s ok.' % chrnames[ichr][3:]
        else:
            print 'Chromosome %s wrong!' % chrnames[ichr][3:]
            nbad += 1
    print 'Testing _get_allchrsizes(): %i / %i good.' % (nchrs - nbad, nchrs)
    print


def test_get_cytoband(pars):
    """
    Read cytoband data for given chromosome and create array
      representation at given resolution.
    """
    cytoband, blimlist, bnamelist, clrlist = rdf._get_cytoband(pars)
    ntrials = 3
    nbad = 0
    # Test band 'chr4    0   4500000 p16.3   gneg'
    if np.allclose(0.0, cytoband[0:90]) and \
            np.allclose(blimlist[0], [0, 90]) and \
            bnamelist[0] == 'p16.3' and \
            clrlist[0] == 'gneg':
                pass
    else:
        print 'Failed to read band:'
        print 'chr4    0   4500000 p16.3   gneg'
        nbad += 1
    # Test band 'chr4    48200000    50400000    p11 acen'
    if np.allclose(cytoband[964:1008], -1.0) and \
            np.allclose(blimlist[11], [964, 1008]) and \
            bnamelist[11] == 'p11' and \
            clrlist[11] == 'acen':
                pass
    else:
        print 'Failed to read band:'
        print 'chr4    48200000    50400000    p11 acen'
        nbad += 1
    # Test band 'chr4    131100000   139500000   q28.3   gpos100'
    if np.allclose(cytoband[2622:2790], 4.0) and \
            np.allclose(blimlist[32], [2622, 2790]) and \
            bnamelist[32] == 'q28.3' and \
            clrlist[32] == 'gpos100':
                pass
    else:
        print 'Failed to read band:'
        print 'chr4    131100000   139500000   q28.3   gpos100'
        nbad += 1
    print 'Testing _get_cytoband(): %i / %i trials ok.' % \
                    (ntrials - nbad, ntrials)
    print


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


################################################

# Retrieving binary archives


def _get_runbinarydir(pars):
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


def _get_runbinarydir_interchr(pars):
    """
    Get directory pointing to binary data archive.
    For inter-chromosomal data.
    """
    rundir = pars['rundir']
    accession = pars['accession']
    runlabel = pars['runlabel']
    chrfullname1 = pars['chrfullname1']
    chrfullname2 = pars['chrfullname2']
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
    else:
        print 'Inter-chromosome data not available!'
        print binarydir
        print 'Run 20_InterChr-Extractor.py ...'
        return
    return fmat, md1, md2


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
        hfio._pickle_securedump(fname, tsetmap, freed=freed)
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


def _update_ConstructMC_20160801(rdata, tsetdata, tsetdatadir, tsetdataprefix,
                chrfullname, region, res, dataset, rhomode='frac'):
    """
    Update database with output from run_ConstructMC_fullarray
    Version 20160801: Split data dicts
    """
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
    # Write to file
    resname = str(res / 1000) + 'kb'
    dirname = os.path.join(tsetdatadir, tsetdataprefix,
                    chrfullname, region, resname)
    rfname = os.path.join(dirname, dataset + '-rhodict' + rhomodesfx + '.p')
    tfname = os.path.join(dirname, dataset + '-tsetdict' + rhomodesfx + '.p')
    hfio._pickle_securedumps((rfname, tfname), (rdict, tdict), freed=False)
    return


################################################

# To deprecate...

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



##############################################################

# Test msmTPT


##############################################################

# Test msmTsetOpt






##############################################################

# Run tests

## readDataFile

genomedatadir = '/home/tanzw/data/genomedata/'

pars = {'genomedatadir': genomedatadir,
        'genomeref': 'hg19',
        'chrref': 'chr4',
        'res': 50000}

test_get_allchrsizes(genomedatadir)
test_get_cytoband(pars)

