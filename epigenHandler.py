#!/usr/bin/env python
"""
Tools for extracting epigenetic data tracks.

Part of ChromaWalker package"""


import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import dataFileReader as dfr
import hiCGraphProc as hgp


################################

epigentypes_default = ['Histone', 'TFBS', 'DNase', 'RNA']
tracktypes_default = ['signal', 'coverage']

class EpigenHandler:
    """
    Wrapper for extracting bigwig / bigbed data tracks from local file or
    UCSC server.
    Note: Access to bigbed coverage tracks on UCSC server not checked yet,
          paths may need to be changed!!
    """
    def __init__(self, pars, epigenpars):
        self.genomedatadir = pars['genomedatadir']
        self.genomeref = pars['genomeref']
        self.epigendatadir = epigenpars['epigendatadir']
        self.cellLine = epigenpars['cellLine']
        self.binsize = epigenpars['binsize']
        self.datalist = epigenpars['datalist']
        self.epigentypes = epigenpars.get('epigentypes', epigentypes_default)
        self.tracktypes = epigenpars.get('tracktypes', tracktypes_default)
        self.epigenURL = 'http://hgdownload.cse.ucsc.edu/gbdb/hg19/bbi/'

    def _get_epigenfilename(self, signal, eptype, source, trtype='signal'):
        """
        Get target file name by searching an index file.
        """
        ## Read data file list
        if eptype in self.epigentypes and trtype in self.tracktypes:
            filetype = '-bigbeds' if trtype == 'coverage' else '-bigwigs'
            flistfname = os.path.join(self.epigendatadir,
                            eptype + filetype + '.txt')
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
        ## Select file
        fnamechoice = None
        fnamebackup = None
        ### If source is not found, use the first track with the same data set
        for sig, src, fname in zip(signals, sources, fnames):
            if sig == signal:
                if src == '' or src == source or \
                            src == ' '.join([source, self.cellLine]):
                    src = source
                    fnamechoice = fname
                    break
                elif fnamebackup is None:
                    fnamebackup = fname
        if fnamechoice is None:
            print 'Using fnamebackup'
            fnamechoice = fnamebackup
        return fnamechoice

    def _get_epigenTrack(self, fname, trtype, cname):
        """
        Get track from file.
        """
        # Get chromosome data / run parameters
        chrlen = dfr._get_allchrsizes([cname], self.genomeref,
                    self.genomedatadir)[0]
        nbins = int(np.ceil(float(chrlen) / self.binsize))
        if trtype == 'coverage':
            cmdname = 'bigBedSummary'
        elif trtype == 'signal':
            cmdname = 'bigWigSummary'
        else:
            print 'Invalid trtype!'
            return None
        cmdstr = [cmdname, fname, cname, str(0), str(nbins * self.binsize),
                str(nbins)]
        ## Execute subprocess to obtain data string
        datalist = subprocess.check_output(cmdstr).split()
        # Parse output data
        for a, v in enumerate(datalist):
            if v == 'n/a':
                datalist[a] = 0.0
            else:
                datalist[a] = float(v)
        return np.array(datalist)

    def _get_epigenPartition(self, fname, trtype, cname, start, end):
        """
        Get partition average from file.
        """
        # Get chromosome data / run parameters
        if trtype == 'coverage':
            cmdname = 'bigBedSummary'
        elif trtype == 'signal':
            cmdname = 'bigWigSummary'
        else:
            print 'Invalid trtype!'
            return None
        cmdstr = [cmdname, fname, cname, str(start), str(end), '1']
        ## Execute subprocess to obtain data string
        data = subprocess.check_output(cmdstr).split()[0]
        # Parse output data
        if data[0].lower() == 'n':
            return 0.0
        else:
            return float(data)

    def _get_alpha(self, cname, signal, eptype, source,
                fname1, trackdata, local=True):
        """
        Get Poisson model noise in bigwig signal.
        """
        trtype = 'signal'
        # Find control data file
        if eptype == 'TFBS' and source == 'SYDH':
            cutindex = fname1.index('Sig.bigWig')
            sigfnamecut = fname1[:cutindex]
            if sigfnamecut.endswith('Std'):
                sig2 = 'ControlStd'
            elif sigfnamecut.endswith('Iggmus'):
                sig2 = 'ControlIggmus'
            elif sigfnamecut.endswith('Iggrab'):
                sig2 = 'ControlIggrab'
            else:
                print 'Can\'t find correct control data!'
                return np.nan
        elif eptype == 'Histone':
            sig2 = 'Control'
        else:
            print 'No control data available!'
            return np.nan
        fnamechoice2 = self._get_epigenfilename(sig2, eptype, source,
                        trtype=trtype)
        fname = os.path.join(self.epigendatadir if local
                    else self.epigenURL, fnamechoice2)
        ctrvec = self._get_epigenTrack(fname, trtype, cname)
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
                alphavals.pop()
                alpha = alphavals[-1]
                break
        return alpha

    def get_epigenTrack(self, cname, signal, eptype, source, trtype='signal',
                getalpha=False, local=True):
        """
        Extract epigenetic data track from bigbed coverage / bigwig signal files.

        'eptype' is the assay type, 'signal' is the factor name,
        'source' is a label for the group that performed the data collection.
        'trtype' should be either "signal" or "coverage", to specify whether to
        read bigwigs or bigbeds.

        Noise model: If getalpha is True, estimate signal-noise parameter
            alpha using a simple Poisson model (only for signal tracks):
                readout = (1-alpha)*signal + alpha*noise

        BigWig data source: If local is True, reads local bigwig file.
            If False, query data from UCSC server.
        """
        # Find file name
        fnamechoice = self._get_epigenfilename(signal, eptype, source,
                        trtype=trtype)
        # Get signal / coverage track
        if local:
            fname = os.path.join(self.epigendatadir, fnamechoice)
        else:
            fname = os.path.join(self.epigenURL, fnamechoice.split('/')[-1])
        trackdata = self._get_epigenTrack(fname, trtype, cname)
        tracklabel = '%s %s (%s)' % (signal, trtype, source)
        # alpha
        if not getalpha:
            return trackdata, tracklabel
        else:
            alpha = self._get_alpha(cname, signal, eptype, source,
                    fnamechoice, trackdata, local=local)
            return trackdata, tracklabel, alpha

    def get_epigenPartition(self, cname, start, end, signal, eptype,
                source, trtype='signal', local=True):
        """
        Extract epigenetic data from bigbed coverage / bigwig signal files.
        Returns average value within a partition.

        'eptype' is the assay type, 'signal' is the factor name,
        'source' is a label for the group that performed the data collection.
        'trtype' should be either "signal" or "coverage", to specify whether to
        read bigwigs or bigbeds.

        BigWig data source: If local is True, reads local bigwig file.
            If False, query data from UCSC server.
        """
        # Find file name
        fnamechoice = self._get_epigenfilename(signal, eptype, source,
                        trtype=trtype)
        # Get signal / coverage track
        if local:
            fname = os.path.join(self.epigendatadir, fnamechoice)
        else:
            fname = os.path.join(self.epigenURL, fnamechoice.split('/')[-1])
        data = self._get_epigenPartition(fname, trtype, cname, start, end)
        label = '%s %s (%s)' % (signal, trtype, source)
        return data, label

    def get_epigenPartitionDataFrame(self, nodedata, local=True):
        """
        Extract epigenetic average data values for each partition. Store data
        in a pandas dataframe.
        """
        epigendata = nodedata[['Position', 'chr', 'st', 'en']].reset_index(
                            drop=True)
        nnodes = len(epigendata)
        for i, data in enumerate(self.datalist):
            signal = data['signal']
            eptype = data['eptype']
            source = data['source']
            trtype = data['trtype']
            vals = []
            print 'Getting data for %s %s (%s)...' % (signal, trtype, source)
            for j in range(nnodes):
                cname, start, end = epigendata.ix[j][['chr', 'st', 'en']]
                val, lbl = self.get_epigenPartition(cname, start, end,
                        signal, eptype, source, trtype=trtype, local=local)
                vals.append(val)
            epigendata[lbl] = vals
        return epigendata

    def get_epigenPartitionDataFrame_ZScore(self, epigendata):
        """
        Extract epigenetic average data values for each partition. Store data
        in a pandas dataframe.
        """
        cols = epigendata.columns
        selcols = cols[(cols!= 'chr') & (cols != 'st') & (cols != 'en')]
        zscoredata = epigendata[['Position']].copy(deep=True)
        weights = np.array(epigendata['en']) - np.array(epigendata['st'])
        for col in selcols:
            if col == 'Position': continue
            col2 = col + ' ZScore'
            vec = epigendata[col]
            zscoredata[col2] = hgp._epigen_vecToZScore_weighted(vec, weights)
        return zscoredata


###################################################

if __name__ == '__main__':
    plt.ion()
    print
    print '**********************************************'
    print 'Welcome to epigenHandler test suite!'
    print '**********************************************'
    print
    cname = 'chr17'
    binsize = 50000
    start = 0
    end = 1000000
    datalist = [{'signal': 'CTCF', 'eptype': 'TFBS',
                 'source': 'SYDH', 'trtype': 'signal'},
                {'signal': 'H3K27Ac', 'eptype': 'Histone',
                 'source': 'Broad', 'trtype': 'signal'},
                {'signal': 'H3K27Me3_2', 'eptype': 'Histone',
                 'source': 'Broad', 'trtype': 'signal'},
                {'signal': 'DNase', 'eptype': 'DNase',
                 'source': 'OpenChrom', 'trtype': 'signal'}
                ]
    pars = {
            'rawdatadir': '/home/tanzw/data/hicdata/ProcessedData/',
            'genomedatadir': 'asciidata/',
            'genomeref': 'hg19',
            'rundir': 'rundata/',
            'accession': 'GSE63525',
            'runlabel': 'GM12878_primary',
            'tsetdatadir': 'rundata/TargetsetOptimization/',
            'tsetdataprefix': 'Full-ConstructMC',
            'reportdir': 'reports/'
            }
    epigenpars = {'epigendatadir': 'epigenomic-tracks',
                  'cellLine': 'GM12878',
                  'binsize': binsize,
                  'datalist': datalist}
    eh = EpigenHandler(pars, epigenpars)
    signal = 'CTCF'
    eptype = 'TFBS'
    source = 'SYDH'
    trtype = 'signal'
    val, lbl = eh.get_epigenPartition(cname, start, end, signal, eptype,
                source, trtype=trtype, local=False)
    print '%s:%i-%i %s: %e' % (cname, start, end, lbl, val)
    track, lbl = eh.get_epigenTrack(cname, signal, eptype, source,
                trtype=trtype, getalpha=False, local=False)
    plt.plot(track)
    plt.suptitle('%s %s' % (cname, lbl))
    _ = raw_input('Enter anything to continue: ')
    plt.close('all')
    f, x = plt.subplots(2, 2, figsize=(12, 6))
    x = x.flatten()
    for i, data in enumerate(datalist):
        signal = data['signal']
        eptype = data['eptype']
        source = data['source']
        trtype = data['trtype']
        track, lbl = eh.get_epigenTrack(cname, signal, eptype, source,
                trtype=trtype, getalpha=False, local=False)
        x[i].plot(track)
        x[i].set_title(lbl)
        x[i].set_ylim(ymin=0)
    f.suptitle('%s epigenetic tracks' % cname)
    _ = raw_input('Enter anything to exit: ')

