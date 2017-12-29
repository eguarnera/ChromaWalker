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
from time import time
import hicutils as hcu
import plotutils as plu
import hicFileIO as hfio
import dataFileReader as dfr
import msmTPT as mt
import msmTsetOpt as mto


class ChromaWalker:

    def __init__(self, pars, epigenpars=None, conMCpars=None, pertpars=None):
        """
        initialize ChromaWalker instance.
        """
        self.DFR= dfr.DataFileReader(pars, epigenpars=epigenpars)
        self.TOpt = mto.TsetOptimizer(pars, DFR=self.DFR,
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

    def getAllFMCmats(self, nproc=1):
        """
        Compute all arrays f/m/cmat for all chromosomes, at the
        highest good beta (less than 10% of loci excluded).
        Also gets inter-chromosomal fmat at beta = 1.0.
        Dumps goodbeta mapping to file.
        To run multiple chromosomes in parallel, set number of
        daughter processes nproc
        """
        pass

    def tsetOptimizerLoop(self, cname, beta=None, interactive=False):
        """
        Run TsetOptimizer in a loop, displaying analytics after each loop.
        If beta is None, optimize at corresponding goodbeta for the chromosome.
        If interactive is True, will prompt user to decide whether to break
        out of loop. Otherwise, Use pre-determined criteria to decide
        when to stop.
        """
        pass

    def autoTsetOptimization(self, nproc=1):
        """
        Automated targetset optimization over all chromosomes.
        To run multiple chromosomes in parallel, set number of
        daughter processes nproc
        """
        pass

    def getGenomeEffectiveNetwork(self, bestmeansize=None):
        """
        Compute whole-genome effective interaction network, choosing
        good/optimal levels of hierarchy on each chromosome with
        mean partition size closet to bestmeansize (in units of Mbp).

        Dumps node data and edge data to Cytoscape CSV and pandas Dataframe.

        Note that to ensure choice of partitions remain consistent, once this
        has been performed you won't be able to recalculate FMCmats or
        further optimize targetsets... If you really want to perform more
        optimization or a technical replicate, I recommend starting afresh
        with a different runlabel.
        """
        pass

    def epigenGenomeEffectiveNetwork(self, bestmeansize=None):
        """
        Compute epigenetic data on partition network. Assumes that the
        network with the same bestmeansize has been computed already.

        Dumps node data to Cytoscape CSV and pandas Dataframe.
        """
        pass


