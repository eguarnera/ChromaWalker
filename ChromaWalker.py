#!/usr/bin/env python
"""
Identifying structural organization of chromatin using Hi-C data analysis.

Main script in ChromaWalker package
"""


import numpy as np
import os
import sys
import matplotlib.pyplot as plt
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


class ChromaWalker:

    def __init__(self, pars, epigenpars=None, conMCpars=None, pertpars=None):
        """
        initialize ChromaWalker instance.
        """
        self.DFR= dfr.DataFileReader(pars, epigenpars=epigenpars)
        self.TOpt = mto.TargetOptimizer(pars, DFR=self.DFR,
                        conMCpars=conMCpars, pertpars=pertpars)

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
            fullsize = len(self.DFR.get_mmat(cname, 1.0))
            # Test all other beta, starting from highest
            blist = np.sort(self.betalist)[::-1]
            for beta in blist:
                print 'Testing beta = %i...' % beta
                thissize = len(self.DFR.get_mmat(cname, beta))
                if thissize >= (1.0 - maxdisconnected) * fullsize:
                    # This beta is good
                    self.bestbeta[cname] = beta
                    break
                else:
                    # This beta is too high, reset array data to dummy values
                    self.DFR._set_dummyarrays(cname, beta)
        return self.bestbeta[cname]

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
        dirname = os.path.join(self.rundir, self.tsetdatadir,
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
                    return True
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
                'nodedata-ms%0e-%s.pkl.gz' % (bestmeansize,
                        'good' if goodLevels else 'optimal'))
        if os.path.isfile(ndfname):
            self.nodedata = pd.read_pickle(ndfname)
        else:
            # Call utility to create DataFrame
            self.nodedata = _get_nodeDataframe(self.cnamelist,
                        self.partitiondata, self.res)
            pd.to_pickle(self.nodedata, ndfname)
        return self.nodedata

    def get_edgeDataframe(self, bestmeansize=1.0, goodLevels=False):
        """
        Get pandas DataFrame of partition edges, either from pickled file or
        from targetset data.
        """
        edfname = os.path.join(self.casedir,
                'edgedata-ms%0e-%s.pkl.gz' % (bestmeansize,
                        'good' if goodLevels else 'optimal'))
        if os.path.isfile(edfname):
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
            fa = np.sum(self.effInteraction, axis=0)
            self.affinity = self.effInteraction / np.outer(fa, fa) - 1.0
            self.edgedata = _get_edgeDataframe(self.nodedata,
                        self.effInteraction, self.affinity)
            pd.to_pickle(self.edgedata, edfname)
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
                        for i, cname in enumerate(cnamelist)}
        #######################
        # Choose levels
        self.bestlevels = _find_bestFitLevels(self.chrsizes,
                    self.goodntargets if goodLevels else self.optimalntargets,
                    bestmeansize / self.res * 1.0e6)
        #######################
        # Assign case index, and update case map file
        print
        print 'Processing case %04i...' % self._get_caseIndex()
        print
        #######################
        # Define partitions: create nodes dataframe
        self.partitiondata = [self.TOpt.get_partitions(cname,
                self.bestbeta[cname], self.bestlevels[cname])
                    for cname in cnamelist]
        _ = self.get_nodeDataframe(bestmeansize=bestmeansize,
                        goodLevels=goodLevels)
        print
        print 'Partition node data sample:'
        print
        print self.nodedata.head()
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

    def epigenGenomeEffectiveNetwork(self, bestmeansize=None):
        """
        Compute epigenetic data on partition network. Assumes that the
        network with the same bestmeansize has been computed already.

        Dumps node data to Cytoscape CSV and pandas Dataframe.

        Note: Not implemented yet.
        """
        pass


#######################################################################


if __name__ == '__main__':
    ################################
    # Perform tests on class functions
    plt.ion()
    print
    print '**********************************************'
    print 'Welcome to ChromaWalker test suite!'
    print '**********************************************'
    print
    baseres = 50000
    res = 50000
    cnamelist = ['chr21', 'chr22']
    betalist = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    norm = 'gfilter_2e5'
    meansize = 1.0
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
            'reportdir': 'reports/',
            'baseres': baseres,
            'res': res,
            'norm': norm,
            'meansize': meansize,
            'cnamelist': cnamelist,
            'betalist': betalist
            }
    epigenpars = {'epigendatadir': 'epigenomic-tracks'}
    cw = ChromaWalker(pars, epigenpars=epigenpars)
    cw.getAllFMCmats()
    cw.autoTsetOptimization()
    cw.getGenomeEffectiveNetwork(bestmeansize=3.0, goodLevels=True)
    #print 'chrsizes:', cw.chrsizes
    #print 'goodlevels:', cw.goodntargets
    #print 'optimallevels:', cw.optimalntargets
    #print 'bestlevels:', cw.bestlevels

