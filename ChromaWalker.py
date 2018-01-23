#!/usr/bin/env python
"""
Identifying structural organization of chromatin using Hi-C data analysis.

Main script in ChromaWalker package
"""


import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec as gridspec
from matplotlib import ticker
import random
import copy
import multiprocessing
import pandas as pd
from time import time
import hicutils as hcu
import plotutils as plu
import hicFileIO as hfio
import dataFileReader as dfr
import msmTPT as mt
import msmTsetOpt as mto
import epigenHandler as eh
import optimals


###############################


def _find_bestFitLevels(chrsizes, chrlevels, bestsize):
    """
    Find levels of hierarchy listed in dict chrlevels that give average
    partition sizes closest to bestsize. chrsizes and bestsize are in pixels.
    """
    chrchoice = {}
    for cname in chrsizes:
        thiscsize = chrsizes[cname]
        thislevels = np.array(chrlevels[cname])
        thispsizes = float(thiscsize) / thislevels
        chrchoice[cname] = thislevels[np.argmin(np.abs(thispsizes - bestsize))]
    return chrchoice


def _get_nodeDataframe(cnamelist, partitiondata, res):
    """
    Create pandas dataframe for partitions as network nodes.
    """
    datarows = []
    ids = []
    chrs = []
    sts = []
    ens = []
    psizes = []
    for cname, (_, thislims, thisids) in zip(cnamelist, partitiondata):
        for thislim, thisid in zip(thislims, thisids):
            if thisid >= 0:
                st, en = thislim * res
                psize = en - st
                datarows.append(('%s:%i-%i' % (cname, st, en),
                                 cname,
                                 st,
                                 en,
                                 psize))
    datacols = ['Position', 'chr', 'st', 'en', 'Partition size']
    nodedata = pd.DataFrame(datarows, columns=datacols)
    return nodedata


def _get_edgeDataframe(nodedata, effInteraction, affinity):
    """
    Create pandas dataframe of effective interactions and affinity between
    partitions.
    """
    # Create DataFrame
    partitionlabels = list(nodedata['Position'])
    if len(partitionlabels) != len(effInteraction):
        print """
Warning: Whole-genome effective interaction matrix not the same size as
     partition node network!
              """
        return
    datarows = []
    for i1, plbl1 in enumerate(partitionlabels):
        for i2, plbl2 in enumerate(partitionlabels):
            datarows.append((plbl1, plbl2,
                effInteraction[i1, i2], affinity[i1, i2]))
    datacols = ['Position1', 'Position2', 'Effective interaction', 'Affinity']
    edgedata = pd.DataFrame(datarows, columns=datacols)
    return edgedata


def _get_bandstainDataframe(nodedata, cytobanddatadict, res):
    """
    Create pandas dataframe of G-stain composition of partitions.
    In units of pixels (in cytobanddatadict).
    """
    stainlevels = [0.0, 1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 0.25]
    staincolors = ['gneg', 'gpos25', 'gpos50', 'gpos75', 'gpos100',
                   'acen', 'gvar', 'stalk']
    nd = nodedata.reset_index(drop=True)
    nnodes = len(nd)
    staind = {s: [] for s in stainlevels}
    for i in range(nnodes):
        cname, st, en = nd.ix[i][['chr', 'st', 'en']]
        st /= res
        en /= res
        thiscband = cytobanddatadict[cname][0]
        bandslice = thiscband[st:en]
        for stain in stainlevels:
            staind[stain].append(np.sum(np.abs(bandslice - stain) < 1.0e-6))
    for stain, clr in zip(stainlevels, staincolors):
        nd[clr] = staind[stain]
    return nd


def _get_fmcmatPlotFigAx(cytobanddata=None, colorbar=False):
    """
    Create figure and axes for f/m/cmat plots.
    """
    if cytobanddata is None:
        if colorbar:
            f = plt.figure(figsize=(10, 8))
            gs = gridspec(1, 2, width_ratios=[1.0, 0.05])
            axmat = f.add_subplot(gs[0])
            axcbar = f.add_subplot(gs[1])
            x = [axmat, axcbar]
        else:
            f, axmat = plt.subplots(figsize=(8, 8))
            x = axmat
    else:
        if colorbar:
            f = plt.figure(figsize=(9.1, 8.5))
            gs = gridspec(2, 3,
                width_ratios=[0.05, 1.0, 0.05], height_ratios=[0.05, 1.0])
            axmat = f.add_subplot(gs[1, 1])
            axcytoh = f.add_subplot(gs[0, 1], sharex=axmat)
            axcytov = f.add_subplot(gs[1, 0], sharey=axmat)
            axcbar = f.add_subplot(gs[1, 2])
            x = [axmat, axcytoh, axcytov, axcbar]
        else:
            f = plt.figure(figsize=(8.45, 8.5))
            gs = gridspec(2, 2,
                width_ratios=[0.05, 1.0], height_ratios=[0.05, 1.0])
            axmat = f.add_subplot(gs[1, 1])
            axcytoh = f.add_subplot(gs[0, 1], sharex=axmat)
            axcytov = f.add_subplot(gs[1, 0], sharey=axmat)
            x = [axmat, axcytoh, axcytov]
    return f, x


def _plot_fmat(fmat, mappingdata, res, cytobanddata=None, colorbar=False,
                title=None):
    """
    Create plot for single-chromosome interaction matrix at base resolution.
    To plot cytobands alongside, include cytobanddata.
    To plot colorbar for fmat values, set colorbar to True.
    """
    f, x = _get_fmcmatPlotFigAx(cytobanddata=cytobanddata, colorbar=colorbar)
    if cytobanddata is None:
        if colorbar:
            axmat, axcbar = x
        else:
            axmat = x
    else:
        if colorbar:
            axmat, axcytoh, axcytov, axcbar = x
        else:
            axmat, axcytoh, axcytov = x
    # Plot log(fmat): Using simple colors.LogNorm gets cmap range truncated...
    vmin = np.log(np.min(fmat[fmat > 0.0])) / np.log(10)
    vmax = np.log(np.percentile(fmat[fmat > 0.0], 99)) / np.log(10)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    size = mappingdata[1] * res / 1.0e6
    fmatpad = plu._build_fullarray(fmat, mappingdata, 0.0)
    img = axmat.matshow(np.log(fmatpad) / np.log(10.0),
                extent=[0, size, size, 0], cmap='afmhot_r',
                vmin=vmin, vmax=vmax)
    axmat.set_aspect(1)
    # Plot cytobands
    if cytobanddata is not None:
        plu._plot_cytobands(cytobanddata, res, axcytoh)
        plu._plot_cytobands_vert(cytobanddata, res, axcytov)
    # Plot colorbar
    if colorbar:
        plt.colorbar(img, cax=axcbar)
    # Draw plot title
    if title is not None:
        plt.suptitle(title, fontsize=14)
    return f, x


def _plot_mmat(mmat, mappingdata, res, cytobanddata=None, colorbar=False,
                title=None):
    """
    Create plot for single-chromosome symmetrized MFPT at base resolution.
    To plot cytobands alongside, include cytobanddata.
    To plot colorbar for fmat values, set colorbar to True.
    """
    f, x = _get_fmcmatPlotFigAx(cytobanddata=cytobanddata, colorbar=colorbar)
    if cytobanddata is None:
        if colorbar:
            axmat, axcbar = x
        else:
            axmat = x
    else:
        if colorbar:
            axmat, axcytoh, axcytov, axcbar = x
        else:
            axmat, axcytoh, axcytov = x
    # Plot mmat: log scale
    mmat2 = (mmat + mmat.T) / 2.0
    vmin = np.min(mmat2[mmat2 > 0.0])
    vmax = np.percentile(mmat2[mmat2 > 0.0], 99)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    size = mappingdata[1] * res / 1.0e6
    fmatpad = plu._build_fullarray(mmat2, mappingdata, 0.0)
    img = axmat.matshow(fmatpad, extent=[0, size, size, 0], norm=norm,
                    cmap='jet')
    axmat.set_aspect(1)
    # Plot cytobands
    if cytobanddata is not None:
        plu._plot_cytobands(cytobanddata, res, axcytoh)
        plu._plot_cytobands_vert(cytobanddata, res, axcytov)
    # Plot colorbar
    if colorbar:
        plt.colorbar(img, cax=axcbar)
    # Draw plot title
    if title is not None:
        plt.suptitle(title, fontsize=14)
    return f, x


def _plot_cmat(cmat, mappingdata, res, cytobanddata=None, colorbar=False,
                title=None):
    """
    Create plot for single-chromosome hitting probability at base resolution.
    To plot cytobands alongside, include cytobanddata.
    To plot colorbar for fmat values, set colorbar to True.
    """
    f, x = _get_fmcmatPlotFigAx(cytobanddata=cytobanddata, colorbar=colorbar)
    if cytobanddata is None:
        if colorbar:
            axmat, axcbar = x
        else:
            axmat = x
    else:
        if colorbar:
            axmat, axcytoh, axcytov, axcbar = x
        else:
            axmat, axcytoh, axcytov = x
    # Plot cmat: power law scale in range [0, 1]
    vmin = 0.0
    vmax = 1.0
    norm = colors.PowerNorm(vmin=vmin, vmax=vmax, gamma=0.2)
    size = mappingdata[1] * res / 1.0e6
    fmatpad = plu._build_fullarray(cmat, mappingdata, 0.0)
    img = axmat.matshow(fmatpad, extent=[0, size, size, 0], norm=norm,
                    cmap='jet')
    axmat.set_aspect(1)
    # Plot cytobands
    if cytobanddata is not None:
        plu._plot_cytobands(cytobanddata, res, axcytoh)
        plu._plot_cytobands_vert(cytobanddata, res, axcytov)
    # Plot colorbar
    if colorbar:
        plt.colorbar(img, cax=axcbar)
    # Draw plot title
    if title is not None:
        plt.suptitle(title, fontsize=14)
    return f, x


class ChromaWalker:

    def __init__(self, pars, epigenpars=None, conMCpars=None, pertpars=None):
        """
        initialize ChromaWalker instance.
        """
        self.DFR= dfr.DataFileReader(pars, epigenpars=epigenpars)
        self.TOpt = mto.TargetOptimizer(pars, DFR=self.DFR,
                        conMCpars=conMCpars, pertpars=pertpars)
        self.EH = eh.EpigenHandler(pars, epigenpars)

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
        self.reportdir = pars['reportdir']

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

        #############################
        # Info about targetset optimization:
        ## Which rho index to use?
        ###    To deprecate
        self.rhomode = pars.get('rhomode', 'frac')
        ## Smallest average partition size in Mbp?
        self.meansize = pars.get('meansize', 0.8)
        ## Partitions smaller than this size should be
        ##    merged with neighbors (in Mbp)
        self.minpartsize = pars.get('minpartsize', 0.5)
        ## Criterion for metastability index rho
        self.rhomax = pars.get('rhomax', 0.8)
        ## Skip first minimum in rho?
        self.skipfirst = pars.get('skipfirst', None)

        #############################
        # Info about Laplacian computation:
        ## Thermal annealing beta of interaction matrix used for
        ##    Laplacian computation
        self.fmatbeta = pars.get('fmatbeta', 1.0)
        ## For binning mode, normalize matrix by partition sizes?
        self.matnorm = pars.get('matnorm', 'sum')
        ## For whole-genome partition network, what's the best
        ##    mean partition size (in Mbp) on each chromosome?
        self.bestmeansize = pars.get('bestmeansize', 1.0)
        ## For whole-genome partition network, look in good rho or optimal rho?
        self.genomenetworkmode = pars.get('genomenetworkmode', 'optimalrho')

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
                'reportdir': self.reportdir,
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
                'rhomode': self.rhomode,
                'meansize': self.meansize,
                'minpartsize': self.minpartsize,
                'rhomax': self.rhomax,
                'skipfirst': self.skipfirst,
                'fmatbeta': self.fmatbeta
                }

        #############################
        # Bookkeeping dictionaries
        ## Dictionary of optimal beta for each chromosome
        self.bestbeta = {c: None for c in self.cnamelist}
        ## Dictionary of good/optimal ntargets for each chromosome
        self.goodntargets = {c: None for c in self.cnamelist}
        self.optimalntargets = {c: None for c in self.cnamelist}

        #############################
        # Data placeholders
        self.nodedata = None
        self.edgedata = None

    def _get_bestBeta(self, cname, maxdisconnected=0.1):
        """
        Get highest integer beta (<= 9) such that less than fraction
        maxdisconnected of all nodes in original network (beta=1.0)
        are disconnected.
        Note: Current implementation sets arrays for too-high beta values to
              dummy ones that consume less disk space.
        """
        if self.bestbeta[cname] is not None:
            return self.bestbeta[cname]
        else:
            # Get beta=1.0 array size
            fullsize = len(self.DFR.get_mappingdata(cname, 1.0)[0])
            # Test all other beta, starting from highest
            blist = np.sort(self.betalist)[::-1]
            for beta in blist:
                print 'Testing beta = %i...' % beta
                thissize = len(self.DFR.get_mappingdata(cname, beta)[0])
                if thissize >= (1.0 - maxdisconnected) * fullsize:
                    # This beta is good
                    self.bestbeta[cname] = beta
                    break
                else:
                    # This beta is too high, reset array data to dummy values
                    self.DFR._set_dummyarrays(cname, beta)
        return self.bestbeta[cname]

    def _plot_allFMCmats(self):
        """
        Plot all FMC matrices at best beta and fmat at beta = 1.0, dump files
        to disk.
        """
        print '*****************************'
        print 'Generating plots for f/m/c matrices at best beta...'
        plotdir = os.path.join(self.reportdir, 'current', 'FMCmats')
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)
        for cname in self.cnamelist:
            print 'Chromosome %s' % cname
            # Get arrays
            fmat, mmat, cmat, mappingdata = self.DFR.get_arrays(cname,
                        self.bestbeta[cname])
            # Get cytobands
            cytobanddata = self.DFR.get_bands(cname)
            t = '%s, $\\beta=%i$: $\\log_{10}(f_{ij})$' % (cname,
                            self.bestbeta[cname])
            f, x = _plot_fmat(fmat, mappingdata, self.res,
                    cytobanddata=cytobanddata, colorbar=True, title=t)
            plotfname = os.path.join(plotdir, 'fmat-%s.pdf' % cname)
            f.savefig(plotfname)
            plt.close(f)
            t = '%s, $\\beta=%i$: $m_{ij}$' % (cname, self.bestbeta[cname])
            f, x = _plot_mmat(mmat, mappingdata, self.res,
                    cytobanddata=cytobanddata, colorbar=True, title=t)
            plotfname = os.path.join(plotdir, 'mmat-%s.pdf' % cname)
            f.savefig(plotfname)
            plt.close(f)
            t = '%s, $\\beta=%i$: $c_{ij}$' % (cname, self.bestbeta[cname])
            f, x = _plot_cmat(cmat, mappingdata, self.res,
                    cytobanddata=cytobanddata, colorbar=True, title=t)
            plotfname = os.path.join(plotdir, 'cmat-%s.pdf' % cname)
            f.savefig(plotfname)
            plt.close(f)
        print 'Check plots in directory %s !' % plotdir
        print
        _ = raw_input('Enter anything to continue: ')

    def getAllFMCmats(self):
        """
        Compute all arrays f/m/cmat for all chromosomes, at the
        highest good beta (less than 10% of loci excluded).
        Also gets inter-chromosomal fmat at beta = 1.0.
        Current implementation doesn't support parallel computation across
        chromosomes.
        TODO: Implement async parallel processing.
        """
        # Run each chromosome in serial mode
        print 'Running intra-chromosomal interaction maps...'
        for cname in self.cnamelist:
            print 'Processing Chromosome %s...' % cname
            bestbeta = self._get_bestBeta(cname)
            print 'Chromosome %s best beta: %i' % (cname, bestbeta)
            _ = self.DFR.get_cmat(cname, bestbeta)
        print 'Running inter-chromosomal interaction maps...'
        for i1, cname1 in enumerate(self.cnamelist):
            print 'Pairs with chromosome %s...' % cname1
            for i2, cname2 in enumerate(self.cnamelist):
                if i1 >= i2:
                    continue
                else:
                    _ = self.DFR.get_fmatMapdata_inter(cname1, cname2)
        # Plot matrices, get user to verify
        self._plot_allFMCmats()
        return self.bestbeta

    def tsetOptimizerLoop(self, cname, beta=None, interactive=False,
                    maxiter=10, minupdate=1):
        """
        Run TsetOptimizer in a loop, displaying analytics after each loop.
        If beta is None, optimize at corresponding goodbeta for the chromosome.
        If interactive is True, will prompt user to decide whether to break
        out of loop. Otherwise, Use pre-determined criteria
        (maxiter, minupdate) to decide when to stop:
            maxiter: maximum number of ConMC/PSnap iterations
            minupdate: If this iterations contain less than this number of
                       updated entries, assume converged.
        """
        if self._check_caseIndexed():
            print ("""
    -----------------------------------------------------
    This case has been indexed! Please do not re-run optimization, otherwise
    downstream data would be corrupted!
    -----------------------------------------------------
                    """)
            return
        # Default parameters for non-interactive optimization
        if beta is None:
            beta = self.bestbeta.get(cname, self._get_bestBeta(cname))
        #################################
        # Run initialization, then ConMC / PSnap loop
        print
        print '*************************'
        print 'Optimizing target sets for Chromosome %s at beta = %i' % (
                    cname, beta)
        print
        # Seed 2-target set?
        rd, td = self.DFR.get_datadicts(cname)
        if self.DFR.readout_datadicts(rd, td, beta, 2) is None:
            print 'ntarget=2 set not found. Seeding...'
            self.TOpt.seed(cname, beta)
        else:
            print 'ntarget=2 set found.'
        # Begin iteration loop
        niter = 0
        while True:
            print
            print '--- Chromosome %s optimization loop %i:' % (cname, niter)
            print
            nupdate = 0
            # ConstructMC
            print 'ConstructMC...'
            nupdate += self.TOpt.conMC(cname, beta)
            # PertSnap
            print 'PertSnap...'
            nupdate += self.TOpt.pSnap(cname, beta)
            # Break iterations?
            if interactive:
                choice = raw_input('Continue iterating? [Y/n]: ')
                if len(choice) > 0 and choice[0].lower() == 'y':
                    continue
                else:
                    print 'Exiting optimization loop...'
                    print
                    break
            niter += 1
            if niter > maxiter or nupdate < minupdate:
                break
                print 'Exiting optimization loop...'
                print

    def _plot_partitionHierarchy(self):
        """
        Generate reports on partition hierarchy defined by good levels, dump
        plots to file.
        """
        print '*****************************'
        print 'Generating plots for partitions...'
        plotdir = os.path.join(self.reportdir, 'current', 'Partitions')
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)
        for cname in self.cnamelist:
            print 'Chromosome %s' % cname
            beta = self.bestbeta[cname]
            cytobanddata = self.DFR.get_bands(cname)
            ngoodlevels = len(self.TOpt.get_goodLevels(cname, beta))
            f = plt.figure(figsize=(12, ngoodlevels * 0.8 + 0.5))
            gs = gridspec(2, 2, height_ratios=[ngoodlevels * 0.8, 0.5])
            axparts = f.add_subplot(gs[0, 1])
            axcytoh = f.add_subplot(gs[1, 1], sharex=axparts)
            axrho = f.add_subplot(gs[0, 0])
            rlist, ntlist = self.TOpt._get_rholist(cname, beta)
            self.TOpt.plot_partitionHierarchy_optimal(axparts, cname, beta,
                        rhomax=self.rhomax, optimal=False)
            plu._plot_cytobands(cytobanddata, self.res, axcytoh)
            axparts.xaxis.set_visible(False)
            axparts.set_ylabel('$n$')
            axrho.plot(ntlist, rlist, '-o')
            axrho.axhline(y=self.rhomax, ls='--', lw=0.5)
            axrho.set_ylim(0, 1)
            axrho.set_ylabel('$\\rho$')
            axrho.set_xlabel('$n$')
            f.suptitle('%s: Good partitioning levels' % cname)
            plotfname = os.path.join(plotdir, 'GoodLevels-%s.pdf' % cname)
            f.savefig(plotfname)
            plt.close(f)
        print 'Check plots in directory %s !' % plotdir
        print
        _ = raw_input('Enter anything to continue: ')

    def autoTsetOptimization(self):
        """
        Automated targetset optimization over all chromosomes.
        Current implementation doesn't support parallel computation across
        chromosomes.
        TODO: Plan/implement async parallel processing.
        """
        if self._check_caseIndexed():
            print ("""
    -----------------------------------------------------
    This case has been indexed! Please do not re-run optimization, otherwise
    downstream data would be corrupted!
    -----------------------------------------------------
                    """)
            return
        print '***********************************'
        print 'Automated targetset optimization...'
        print '***********************************'
        print
        for cname in self.cnamelist:
            self.tsetOptimizerLoop(cname)
        print
        print 'Automated targetset optimization done!'
        print '***********************************'
        print
        # Show partitioning hierarchy of all good levels
        self._plot_partitionHierarchy()

    def getAllGoodLevels(self):
        """
        Find all good levels of structural hierarchy (ntargets)
        for each chromosome.
        """
        for cname in self.cnamelist:
            beta = self.bestbeta.get(cname, self._get_bestBeta(cname))
            self.goodntargets[cname] = self.TOpt.get_goodLevels(cname, beta)
            if len(self.goodntargets[cname]) == 0:
                print 'Warning: Chromosome ' + cname + \
                        ' has no good levels of hierarchy!'

    def getAllOptimalLevels(self):
        """
        Find all optimal levels of structural hierarchy (ntargets)
        for each chromosome.
        """
        for cname in self.cnamelist:
            beta = self.bestbeta.get(cname, self._get_bestBeta(cname))
            self.optimalntargets[cname] = self.TOpt.get_optimalLevels(
                                cname, beta)
            if len(self.optimalntargets[cname]) == 0:
                print 'Warning: Chromosome ' + cname + \
                        ' has no optimal levels of hierarchy!'

    def _check_caseIndexed(self):
        """
        Check if a particular test case has been indexed.
        """
        # Get / create case mapping file
        basedir = os.path.join(self.rundir, 'ChromaWalker')
        if not os.path.isdir(basedir):
            os.makedirs(basedir)
        mapfname = os.path.join(basedir, 'casemap.p')
        thispar = copy.deepcopy(self.basepars)
        resname = str(self.res / 1000) + 'kb'
        rhomodesfx = mt._get_rhomodesfx(self.rhomode)
        dirname = os.path.join(self.tsetdatadir,
                self.tsetdataprefix, self.cnamelist[-1], self.region, resname)
        dataset = dfr._get_tsetdataset2(thispar)
        rdatafname = os.path.join(dirname, dataset + '-rhodict' +
                        rhomodesfx + '.p')
        ## Check if the datadict exists
        if not os.path.isfile(rdatafname):
            print ('No corresponding datadicts present:' + rdatafname)
            return None
        if os.path.isfile(mapfname):
            casemap = hfio._pickle_secureread(mapfname, free=False)
            # Find case
            for key, fname in casemap.iteritems():
                if fname == rdatafname:
                    hfio._pickle_secureunlock(mapfname)
                    return True
            hfio._pickle_secureunlock(mapfname)
        return False

    def _get_caseIndex(self):
        """
        Get case index entry in case map file. Create one if not found.
        Note that this prevents users from running tsetOptimizerLoop and
        autoTsetOptimization using ChromaWalker on the same case.

        Each case is tied to the rhodict/tsetdict file assocaited with the
        last chromosome in cnamelist.
        """
        # Get / create case mapping file
        basedir = os.path.join(self.rundir, 'ChromaWalker')
        if not os.path.isdir(basedir):
            os.makedirs(basedir)
        mapfname = os.path.join(basedir, 'casemap.p')
        thispar = copy.deepcopy(self.basepars)
        resname = str(self.res / 1000) + 'kb'
        rhomodesfx = mt._get_rhomodesfx(self.rhomode)
        dirname = os.path.join(self.tsetdatadir,
                self.tsetdataprefix, self.cnamelist[-1], self.region, resname)
        dataset = dfr._get_tsetdataset2(thispar)
        rdatafname = os.path.join(dirname, dataset + '-rhodict' +
                        rhomodesfx + '.p')
        ## Check if the datadict exists
        if not os.path.isfile(rdatafname):
            print ('No corresponding datadicts present:' +
                    'Please run tsetOptimizerLoop or autoTsetOptimization!')
            return None
        if os.path.isfile(mapfname):
            casemap = hfio._pickle_secureread(mapfname, free=False)
            # Find case
            for key, fname in casemap.iteritems():
                if fname == rdatafname:
                    hfio._pickle_secureunlock(mapfname)
                    self.casedir = os.path.join(basedir, '%04i' % key)
                    return key
            # If none match, create new entry
            key = len(casemap.keys())
            casemap[key] = rdatafname
            hfio._pickle_securedump(mapfname, casemap, freed=False)
        else:
            casemap = {0: rdatafname}
            key = 0
            hfio._pickle_securedump(mapfname, casemap, freed=True)
        # Create case directory
        self.casedir = os.path.join(basedir, '%04i' % key)
        os.makedirs(self.casedir)
        return key

    def get_nodeDataframe(self, bestmeansize=1.0, goodLevels=False):
        """
        Get pandas DataFrame of partition nodes, either from pickled file or
        from targetset data.
        """
        # Test if pickled data already exists
        ndfname = os.path.join(self.casedir,
                'nodedata-ms%.e-%s.pkl.gz' % (bestmeansize,
                        'good' if goodLevels else 'optimal'))
        ndcsv = os.path.join(self.casedir,
                'nodedata-ms%.e-%s.csv' % (bestmeansize,
                        'good' if goodLevels else 'optimal'))
        if os.path.isfile(ndfname):
            print 'Reading node dataframe...'
            self.nodedata = pd.read_pickle(ndfname)
            #raw_input('Reading nodeDF: ')
        else:
            # Call utility to create DataFrame
            #raw_input('Trying to create nodeDF: ')
            self.nodedata = _get_nodeDataframe(self.cnamelist,
                        self.partitiondata, self.res)
            pd.to_pickle(self.nodedata, ndfname)
            self.nodedata.to_csv(ndcsv, index=False)
        return self.nodedata

    def get_nodeBandDataframe(self, bestmeansize=1.0, goodLevels=False):
        """
        Get pandas DataFrame of partition nodes, either from pickled file or
        from targetset data.
        """
        # Test if pickled data already exists
        ndfname = os.path.join(self.casedir,
                'nodedata-bands-ms%.e-%s.pkl.gz' % (bestmeansize,
                        'good' if goodLevels else 'optimal'))
        ndcsv = os.path.join(self.casedir,
                'nodedata-bands-ms%.e-%s.csv' % (bestmeansize,
                        'good' if goodLevels else 'optimal'))
        if os.path.isfile(ndfname):
            print 'Reading node dataframe...'
            self.nodebanddata = pd.read_pickle(ndfname)
        else:
            self.nodedata = self.get_nodeDataframe(bestmeansize=bestmeansize,
                    goodLevels=goodLevels)
            # Call utility to create DataFrame
            cbanddatadict = {cname: self.DFR.get_bands(cname)
                             for cname in self.cnamelist}
            self.nodebanddata = _get_bandstainDataframe(self.nodedata,
                            cbanddatadict, self.res)
            pd.to_pickle(self.nodebanddata, ndfname)
            self.nodebanddata.to_csv(ndcsv, index=False)
        return self.nodebanddata

    def get_nodeEpigenDataframe(self, bestmeansize=1.0, goodLevels=False):
        """
        Get pandas DataFrame of partition nodes (epigenetic signal data).
        """
        # Test if pickled data already exists
        ndfname = os.path.join(self.casedir,
                'nodedata-epigen-ms%.e-%s.pkl.gz' % (bestmeansize,
                        'good' if goodLevels else 'optimal'))
        ndcsv = os.path.join(self.casedir,
                'nodedata-epigen-ms%.e-%s.csv' % (bestmeansize,
                        'good' if goodLevels else 'optimal'))
        if os.path.isfile(ndfname):
            print 'Reading node epigen dataframe...'
            self.nodeepigendata = pd.read_pickle(ndfname)
        else:
            self.nodedata = self.get_nodeDataframe(bestmeansize=bestmeansize,
                    goodLevels=goodLevels)
            # Call utility to create DataFrame
            print
            print 'Getting epigenetic track data from UCSC server...'
            print 'This might take a while.'
            print
            self.nodeepigendata = self.EH.get_epigenPartitionDataFrame(
                        self.nodedata, local=False)
            pd.to_pickle(self.nodeepigendata, ndfname)
            self.nodeepigendata.to_csv(ndcsv, index=False)
        return self.nodeepigendata

    def get_nodeEpigenDataframe_ZScore(self, bestmeansize=1.0,
                    goodLevels=False):
        """
        Get pandas DataFrame of partition nodes (epigenetic signal Z-Scores).
        """
        # Test if pickled data already exists
        ndfname = os.path.join(self.casedir,
                'nodedata-epigenZ-ms%.e-%s.pkl.gz' % (bestmeansize,
                        'good' if goodLevels else 'optimal'))
        ndcsv = os.path.join(self.casedir,
                'nodedata-epigenZ-ms%.e-%s.csv' % (bestmeansize,
                        'good' if goodLevels else 'optimal'))
        if os.path.isfile(ndfname):
            print 'Reading node epigenZ dataframe...'
            self.nodeepigenz = pd.read_pickle(ndfname)
        else:
            self.nodeepigendata = self.get_nodeEpigenDataframe(
                        bestmeansize=bestmeansize, goodLevels=goodLevels)
            self.nodeepigenz = self.EH.get_epigenPartitionDataFrame_ZScore(
                        self.nodeepigendata)
            pd.to_pickle(self.nodeepigenz, ndfname)
            self.nodeepigenz.to_csv(ndcsv, index=False)
        return self.nodeepigenz

    def get_edgeDataframe(self, bestmeansize=1.0, goodLevels=False):
        """
        Get pandas DataFrame of partition edges, either from pickled file or
        from targetset data.
        """
        edfname = os.path.join(self.casedir,
                'edgedata-ms%.e-%s.pkl.gz' % (bestmeansize,
                        'good' if goodLevels else 'optimal'))
        edcsv = os.path.join(self.casedir,
                'edgedata-ms%.e-%s.csv' % (bestmeansize,
                        'good' if goodLevels else 'optimal'))
        if os.path.isfile(edfname):
            print 'Reading edge dataframe...'
            self.edgedata = pd.read_pickle(edfname)
        else:
            self.get_nodeDataframe(bestmeansize=bestmeansize,
                        goodLevels=goodLevels)
            # Obtain whole-genome effective interaction matrix
            sublmats = [[0 for c1 in self.cnamelist] for c2 in self.cnamelist]
            for i1, cname1 in enumerate(self.cnamelist):
                beta1 = self.bestbeta[cname1]
                ntarget1 = self.bestlevels[cname1]
                for i2, cname2 in enumerate(self.cnamelist):
                    beta2 = self.bestbeta[cname2]
                    ntarget2 = self.bestlevels[cname2]
                    if i1 == i2:
                        sublmats[i1][i2] = self.TOpt.get_binLaplacian(
                                    cname1, beta1, ntarget1)
                    elif i1 < i2:
                        sublmats[i1][i2] = self.TOpt.get_binLaplacian_inter(
                                    cname1, cname2, beta1, beta2,
                                    ntarget1, ntarget2)
                    else:
                        sublmats[i1][i2] = sublmats[i2][i1].T
            self.effInteraction = np.array(np.bmat(sublmats))
            prob = self.effInteraction - np.diag(np.diag(self.effInteraction))
            prob /= np.sum(prob)
            fa = np.sum(prob, axis=0)
            self.affinity = prob / np.outer(fa, fa) - 1.0 + \
                            np.diag(np.ones_like(fa))
            self.edgedata = _get_edgeDataframe(self.nodedata,
                        self.effInteraction, self.affinity)
            pd.to_pickle(self.edgedata, edfname)
            self.edgedata.to_csv(edcsv, index=False)
        return self.edgedata

    def getGenomeEffectiveNetwork(self, bestmeansize=1.0, goodLevels=False):
        """
        Compute whole-genome effective interaction network, choosing
        good/optimal levels of hierarchy on each chromosome with
        mean partition size closet to bestmeansize (in units of Mbp).
        If goodLevels is False, consider only optimal levels of hierarchy.

        Dumps node data and edge data to Cytoscape CSV and pandas Dataframe.

        Note that to ensure choice of partitions remain consistent, once this
        has been performed you won't be able to recalculate FMCmats or
        further optimize targetsets... If you really want to perform more
        optimization, or a technical replicate, I recommend starting afresh
        with a different runlabel (use a symbolic link to read from the same
        HiC data repo) or rundir.
        """
        #######################
        # Find suitable levels
        self.getAllGoodLevels()
        self.getAllOptimalLevels()
        # Get chromosome sizes
        csizes = dfr._get_allchrsizes(self.cnamelist, self.genomeref,
                        self.genomedatadir)
        self.chrsizes = {cname: int(np.ceil(csizes[i] / self.res))
                        for i, cname in enumerate(self.cnamelist)}
        #######################
        # Choose levels
        self.bestlevels = _find_bestFitLevels(self.chrsizes,
                    self.goodntargets if goodLevels else self.optimalntargets,
                    bestmeansize / self.res * 1.0e6)
        #######################################################
        #######################################################
        # Intercept pipeline: Use selected levels defined in optimals
        #print 'Best levels:', self.bestlevels
        #key = self.accession, self.runlabel, False, self.norm
        #self.bestlevels = {cname:
            #optimals.bestbetant[key][cname][1][optimals.fullgenomelevels[key][cname]]
            #for cname in self.cnamelist}
        #print 'Best levels:', self.bestlevels
        #_ = raw_input('...: ')
        #######################################################
        #######################################################
        #######################
        # Assign case index, and update case map file
        print
        print 'Processing case %04i...' % self._get_caseIndex()
        print
        #######################
        # Define partitions: create nodes dataframe
        #self.partitiondata = [self.TOpt.get_partitions(cname,
                #self.bestbeta[cname], self.bestlevels[cname])
                    #for cname in self.cnamelist]
        _ = self.get_nodeDataframe(bestmeansize=bestmeansize,
                        goodLevels=goodLevels)
        print
        print 'Partition node data sample:'
        print
        print self.nodedata.head()
        print
        #######################
        # Get node band dataframe
        _ = self.get_nodeBandDataframe(bestmeansize=bestmeansize,
                        goodLevels=goodLevels)
        print
        print 'Partition node bands data sample:'
        print
        print self.nodebanddata.head()
        print
        #######################
        # Get Laplacians: create edges dataframe
        _ = self.get_edgeDataframe(bestmeansize=bestmeansize,
                        goodLevels=goodLevels)
        print
        print 'Partition edge data sample:'
        print
        print self.edgedata.head()
        print
        #######################
        # Get epigen levels: create epigen dataframe
        _ = self.get_nodeEpigenDataframe(bestmeansize=bestmeansize,
                        goodLevels=goodLevels)
        print
        print 'Partition epigen data sample:'
        print
        print self.nodeepigendata[self.nodeepigendata.columns[:10]].head()
        print
        #######################
        # Get epigen scores: create epigen scores dataframe
        _ = self.get_nodeEpigenDataframe_ZScore(bestmeansize=bestmeansize,
                        goodLevels=goodLevels)
        print
        print 'Partition epigenZ data sample:'
        print
        print self.nodeepigenz[self.nodeepigenz.columns[:10]].head()
        print
        #######################
        # Plor effective interaction and affinity matrices
        self._reportGenomeEffectiveNetwork()

    def _get_partitionNetworkMatrix_1chr(self, cname,
            edgecolumn='Effective interaction'):
        """
        Get data required to plot matrix properties of partition network.
        Single-chromosome version.
        Accepted edgecolumn values:
            - Effective interaction
            - Affinity
        """
        # Extract nodes and edges in intra-chr network
        nodemask = (self.nodedata['chr'] == cname)
        edgemask = (self.edgedata['Position1'].str.startswith('%s:' % cname) &
                    self.edgedata['Position2'].str.startswith('%s:' % cname))
        nd = self.nodedata[nodemask].reset_index()
        ed = self.edgedata[edgemask]
        # Define mapping from matrix to bin index, and bin edges
        bset = list(set(list(nd['st']) + list(nd['en']) + [0]))
        bset.sort()
        bvals = np.array(bset) / 1.0e6
        nbins = len(bset)
        nrows = len(nd)
        binmap = [bset.index(nd.ix[i]['st']) for i in range(nrows)]
        # Define effective interaction matrix
        mat = np.zeros((nbins, nbins))
        for i in range(nrows):
            pos1 = nd.ix[i]['Position']
            mask1 = (ed['Position1'] == pos1)
            for j in range(nrows):
                pos2 = nd.ix[j]['Position']
                mask2 = (ed['Position2'] == pos2)
                if i > j:
                    mat[binmap[i], binmap[j]] = mat[binmap[j], binmap[i]]
                else:
                    mat[binmap[i], binmap[j]] = list(ed[(mask1 & mask2)]
                                [edgecolumn])[0]
        return bvals, mat

    def _get_partitionNetworkMatrix_Nchr(self, cnamelist,
            edgecolumn='Effective interaction'):
        """
        Get data required to plot matrix properties of partition network.
        Multi-chromosome version.
        Accepted edgecolumn values:
            - Effective interaction
            - Affinity
        """
        # Get node data in order of chromosomes
        nd_all = None
        bvals = []
        binmap = []
        chrmidpts = []
        chrbounds = [0.0]
        for cname in cnamelist:
            nodemask = (self.nodedata['chr'] == cname)
            nd = self.nodedata[nodemask].reset_index()
            if nd_all is None:
                nd_all = nd.copy(deep=True)
            else:
                nd_all = nd_all.append(nd)
            nrow = len(nd)
            bset = list(set(list(nd['st']) + list(nd['en']) + [0]))
            bset.sort()
            bval = np.array(bset) / 1.0e6
            if len(bvals) == 0:
                chrmidpts.append(np.max(bval) / 2.0)
                bvals.extend(list(bval))
                bm = [bset.index(nd.ix[i]['st']) for i in range(nrow)]
                binmap.extend(bm)
            else:
                nbins = len(bvals) - 1
                chrmidpts.append(np.max(bval) / 2.0 + np.max(bvals))
                bvals.extend(list(bval[1:] + np.max(bvals)))
                bm = [bset.index(nd.ix[i]['st']) + nbins for i in range(nrow)]
                binmap.extend(bm)
            chrbounds.append(np.max(bvals))
        # Get edge data (not necessarily in order)
        edgemask = None
        for cname1 in cnamelist:
            m1 = self.edgedata['Position1'].str.startswith('%s:' % cname1)
            for cname2 in cnamelist:
                m2 = self.edgedata['Position2'].str.startswith('%s:' % cname2)
                if edgemask is None:
                    edgemask = m1 & m2
                else:
                    edgemask = edgemask | (m1 & m2)
        ed_all = self.edgedata[edgemask]
        # Define effective interaction matrix
        nbins = len(bvals)
        nrows = len(nd_all)
        nd_all = nd_all.reset_index()
        mat = np.zeros((nbins, nbins))
        poslist = list(nd_all['Position'])
        p1list = list(ed_all['Position1'])
        p2list = list(ed_all['Position2'])
        j1 = np.array([binmap[poslist.index(pos1)] for pos1 in p1list])
        j2 = np.array([binmap[poslist.index(pos2)] for pos2 in p2list])
        mat[j1, j2] = np.array(ed_all[edgecolumn])
        mat += mat.T - np.diag(np.diag(mat))
        #for i in range(len(ed_all)):
            #pos1, pos2, val = ed_all.ix[i][['Position1', 'Position2',
                            #edgecolumn]]
            #j1 = poslist.index(pos1)
            #j2 = poslist.index(pos2)
            #mat[binmap[j1], binmap[j2]] = val
            #mat[binmap[j2], binmap[j1]] = val
        #for i in range(nrows):
            #pos1 = nd_all.ix[i]['Position']
            #mask11 = (ed_all['Position1'] == pos1)
            #for j in range(nrows):
                #pos2 = nd_all.ix[j]['Position']
                #mask21 = (ed_all['Position2'] == pos1)
                #mask22 = (ed_all['Position2'] == pos2)
                #mask12 = (ed_all['Position1'] == pos2)
                #if np.sum(mask11 & mask22) > 0:
                    #mat[binmap[i], binmap[j]] = list(ed_all[(mask11 & mask22)]
                                #[edgecolumn])[0]
                #else:
                    #mat[binmap[i], binmap[j]] = list(ed_all[(mask12 & mask21)]
                                #[edgecolumn])[0]
        return bvals, chrmidpts, chrbounds, mat

    def _plot_effectiveInteraction_intra(self, cname, axis,
                    norm=True, colorbar=False, diagOn=False,
                    scale='pow', n=1.0):
        """
        Plot effective interaction matrix for single chromosome.

        To include self-interactions in the plot, set diagOn to True.

        To toggle between power-law scaling and log scaling of matrix values,
            set scale = 'pow' or 'log'. For power-law scaling, set exponent n.
        """
        bvals, fabmat = self._get_partitionNetworkMatrix_1chr(cname,
            edgecolumn='Effective interaction')
        if not diagOn:
            fabmat -= np.diag(np.diag(fabmat))
        if norm:
            fabmat /= np.sum(fabmat)
        vmin = np.min(fabmat[fabmat > 0.0])
        vmax = np.max(fabmat[fabmat > 0.0])
        # Draw pcolormesh
        if scale == 'pow':
            thisnorm = colors.PowerNorm(gamma=n)
        elif scale == 'log':
            thisnorm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            thisnorm = None
        pcm = axis[0].pcolormesh(bvals, bvals, fabmat, norm=thisnorm,
                       cmap='afmhot_r')
        if colorbar:
            plt.colorbar(pcm, cax=axis[1])
        axis[0].set_xlim(np.min(bvals), np.max(bvals))
        axis[0].set_ylim(np.max(bvals), np.min(bvals))
        axis[0].xaxis.tick_top()
        axis[0].set_aspect(1)

    def _plot_effectiveInteraction_inter(self, cnamelist, axis,
                    norm=True, colorbar=False, diagOn=False,
                    scale='pow', n=1.0):
        """
        Plot effective interaction matrix for multiple chromosomes.

        To include self-interactions in the plot, set diagOn to True.

        To toggle between power-law scaling and log scaling of matrix values,
            set scale = 'pow' or 'log'. For power-law scaling, set exponent n.
        """
        bvals, chrmidpts, chrbounds,  fabmat = \
                self._get_partitionNetworkMatrix_Nchr(cnamelist,
                    edgecolumn='Effective interaction')
        if not diagOn:
            fabmat -= np.diag(np.diag(fabmat))
        if norm:
            fabmat /= np.sum(fabmat)
        vmin = np.min(fabmat[fabmat > 0.0])
        vmax = np.max(fabmat[fabmat > 0.0])
        # Draw pcolormesh
        if scale == 'pow':
            thisnorm = colors.PowerNorm(gamma=n)
        elif scale == 'log':
            thisnorm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            thisnorm = None
        cmap = plt.get_cmap('afmhot_r')
        cmap.set_bad((0.5, 0.5, 0.5))
        fabmat[fabmat == 0.0] = np.nan
        pcm = axis.pcolormesh(bvals, bvals, fabmat, norm=thisnorm,
                       cmap=cmap, vmin=0.0)
        for cb in chrbounds:
            axis.axhline(y=cb, lw=1.0, c='k')
            axis.axvline(x=cb, lw=1.0, c='k')
        axis.set_xticks(chrmidpts)
        axis.set_xticklabels([c[3:] for c in cnamelist])
        axis.set_yticks(chrmidpts)
        axis.set_yticklabels([c[3:] for c in cnamelist])
        if colorbar:
            #cb = plt.colorbar(pcm, ax=axis, format='%.0e')
            def fmt(x, pos):
                a, b = '{:.0e}'.format(x).split('e')
                b = int(b)
                return r'{0}e{1}'.format(a, b)
            cb = plt.colorbar(pcm, ax=axis, format=ticker.FuncFormatter(fmt))
            #tick_locator = ticker.MaxNLocator(nbins=6)
            #cb.locator = tick_locator
            #cb.update_ticks()
        axis.set_xlim(np.min(bvals), np.max(bvals))
        axis.set_ylim(np.max(bvals), np.min(bvals))
        axis.xaxis.tick_top()
        axis.set_aspect(1)

    def _plot_affinity_inter(self, cnamelist, axis,
                    colorbar=False):
        """
        Plot affinity matrix for multiple chromosomes.

        Uses color map bwr in range [-1, 1]
        """
        bvals, chrmidpts, chrbounds,  aabmat = \
                self._get_partitionNetworkMatrix_Nchr(cnamelist,
                    edgecolumn='Affinity')
        aabmat[aabmat == 0.0] = np.nan
        # Draw pcolormesh
        cmap = plt.get_cmap('bwr')
        cmap.set_bad((0.5, 0.5, 0.5))
        pcm = axis.pcolormesh(bvals, bvals, aabmat,
                        vmin=-1, vmax=1, cmap=cmap)
        for cb in chrbounds:
            axis.axhline(y=cb, lw=1.0, c='k')
            axis.axvline(x=cb, lw=1.0, c='k')
        axis.set_xticks(chrmidpts)
        axis.set_xticklabels([c[3:] for c in cnamelist])
        axis.set_yticks(chrmidpts)
        axis.set_yticklabels([c[3:] for c in cnamelist])
        if colorbar:
            cb = plt.colorbar(pcm, ax=axis, format='%i')
            cb.set_ticks([-1, 1])
        axis.set_xlim(np.min(bvals), np.max(bvals))
        axis.set_ylim(np.max(bvals), np.min(bvals))
        axis.xaxis.tick_top()
        axis.set_aspect(1)

    def _reportGenomeEffectiveNetwork(self):
        """
        Generate plots of effective interaction / affinity data.
        """
        plotdir = os.path.join(self.reportdir, 'current', 'WholeGenome')
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)
        f, x = plt.subplots(1, 1, figsize=(10, 8))
        self._plot_effectiveInteraction_inter(self.cnamelist, x,
                    norm=True, colorbar=True, diagOn=False,
                    scale='pow', n=0.2)
        #f.suptitle('Whole-genome normalized effective interactions',
                    #fontsize=14)
        x.tick_params(axis='both', which='major', labelsize=6)
        fname = os.path.join(plotdir, 'EffectiveInteractions.pdf')
        f.savefig(fname)
        plt.close(f)
        f, x = plt.subplots(1, 1, figsize=(10, 8))
        self._plot_affinity_inter(self.cnamelist, x, colorbar=True)
        #f.suptitle('Whole-genome affinity', fontsize=14)
        x.tick_params(axis='both', which='major', labelsize=6)
        fname = os.path.join(plotdir, 'Affinity.pdf')
        f.savefig(fname)
        plt.close(f)
        print 'Check plots in directory %s !' % plotdir
        print
        _ = raw_input('Enter anything to continue: ')


#######################################################################


if __name__ == '__main__':
    ################################
    # Perform tests on class functions
    #plt.ion()
    print
    print '**********************************************'
    print 'Welcome to ChromaWalker test suite!'
    print '**********************************************'
    print
    baseres = 500000
    res = 4000000
    cnamelist = ['chr21', 'chr22']
    cnamelist = ['chr%i' % i for i in range(1, 23)] + ['chrX']
    betalist = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    norm = 'raw'
    meansize = 5.0
    rhomax = 0.8
    datalist = [
                {'signal': 'CTCF',          'eptype': 'TFBS',
                 'source': 'SYDH',          'trtype': 'signal'},
                {'signal': 'H3K27Ac',       'eptype': 'Histone',
                 'source': 'Broad',         'trtype': 'signal'},
                {'signal': 'H3K27Me3_2',    'eptype': 'Histone',
                 'source': 'Broad',         'trtype': 'signal'},
                {'signal': 'DNase',         'eptype': 'DNase',
                 'source': 'OpenChrom',     'trtype': 'signal'}
                ]
    pars = {
            'rawdatadir': '/home/tanzw/data/hicdata/ProcessedData/',
            #'genomedatadir': '/home/tanzw/data/genomedata/',
            #'rawdatadir': 'asciidata/',
            'genomedatadir': 'asciidata/',
            'genomeref': 'hg19',
            'rundir': 'rundata/',
            'accession': 'GSE63525',
            'runlabel': 'GM12878_primary',
            'tsetdatadir': 'rundata/TargetsetOptimization/',
            'tsetdataprefix': 'Full-ConstructMC',
            'reportdir': 'reports/',
            'baseres': baseres,
            'res': res,
            'norm': norm,
            'meansize': meansize,
            'cnamelist': cnamelist,
            'betalist': betalist,
            'rhomax': rhomax
            }
    epigenpars = {
                  #'epigendatadir': 'epigen',                # Sample local files
                  'epigendatadir': 'epigenomic-tracks',
                  'cellLine': 'GM12878',
                  'binsize': res,
                  'datalist': datalist
                 }
    ############################################
    ### Create CW instance
    cw = ChromaWalker(pars, epigenpars=epigenpars)
    ############################################
    ### Convert interaction matrices to binary files, and compute MFPT
    ###  and hitting probability matrices
    cw.getAllFMCmats()
    ############################################
    ### Automated targetset optimization
    ###  For more user control, use method tsetOptimizerLoop
    #cw.autoTsetOptimization()
    ############################################
    ### Define effective interaction network by choosing good/optimal levels
    ###  such that average partition size is closest to bestmeansize (in Mbp).
    ###  See docs for more details.
    #cw.getGenomeEffectiveNetwork(bestmeansize=5.0, goodLevels=True)
    _ = raw_input('Run completed. Enter anything to exit: ')
    print
    print 'Farewell!'
    print

