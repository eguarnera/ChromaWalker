
# Module with functions for partitioning comparison metrics

import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema


def _get_testtruepoll(testpart, truepart):
    """
    Get poll for matches between test/true partitionings.
    """
    testtruepoll = {ipart: {jpart: 0.0 for jpart in truepart}
                    for ipart in testpart}
    for jpart, (stj, enj) in truepart.iteritems():
        for ipart, (sti, eni) in testpart.iteritems():
            if stj >= eni or enj <= sti:
                continue
            testtruepoll[ipart][jpart] = min(eni, enj) - max(sti, stj)
    return testtruepoll


def _print_testtruepoll(testtruepoll):
    """
    Print poll results for test/true partitionings.
    """
    print '\t',
    for jpart in truepart:
        print ('j=%i\t' % jpart),
    print
    for ipart in testpart:
        print 'i=%i\t',
        for jpart in truepart:
            print ('%i\t' % testtruepoll[ipart][jpart]),
        print


def _get_probabilityNorm(testpart, truepart, probnorm):
    nval = 0
    if probnorm == 'full':
        for ipart, (st, en) in testpart.iteritems():
            nval = max(en, nval)
        for jpart, (st, en) in truepart.iteritems():
            nval = max(en, nval)
    elif probnorm == 'true':
        for jpart, (st, en) in truepart.iteritems():
            nval += en - st
    elif probnorm == 'test':
        for ipart, (st, en) in testpart.iteritems():
            nval += en - st
    return nval


def _get_TF_PN(testpart, truepart):
    """
    Get True positive, False positive, False negative matches.
    True negative hard to define unless testpart and truepart have the same set.
    """
    TP = FP = FN = 0.0
    for jpart, (stj, enj) in truepart.iteritems():
        truepartsize = enj - stj
        for ipart, (sti, eni) in testpart.iteritems():
            overlap = 0.0 if (sti > enj or stj > eni) else \
                             (min(eni, enj) - max(sti, stj))
            TP += overlap * (overlap - 1) / 2.0
            FN += overlap * (truepartsize - overlap) / 2.0
    for ipart, (sti, eni) in testpart.iteritems():
        testpartsize = eni - sti
        for jpart, (stj, enj) in truepart.iteritems():
            overlap = 0.0 if (sti > enj or stj > eni) else \
                             (min(eni, enj) - max(sti, stj))
            FP += overlap * (testpartsize - overlap) / 2.0
    return TP, FP, FN


def purity(testpart, truepart):
    """
    Calculate purity measure.
    """
    testtruepoll = _get_testtruepoll(testpart, truepart)
    pollsum = 0.0
    nptssum = 0.0
    purityI = {}
    for ipart, (st, en) in testpart.iteritems():
        npts = float(en - st)
        nptssum += npts
        pollmax = 0
        for jpart in truepart:
            if testtruepoll[ipart][jpart] > pollmax:
                pollmax = testtruepoll[ipart][jpart]
        pollsum += pollmax
        purityI[ipart] = pollmax / npts
    purity = pollsum / nptssum
    return purity, purityI


def FMeasure(testpart, truepart):
    """
    Calculate F-Measure.
    """
    testtruepoll = _get_testtruepoll(testpart, truepart)
    precisionI = {}
    recallI = {}
    for ipart, (st, en) in testpart.iteritems():
        npts = float(en - st)
        pollmax = 0
        for jpart in truepart:
            if testtruepoll[ipart][jpart] > pollmax:
                jbest = jpart
                pollmax = testtruepoll[ipart][jpart]
        precisionI[ipart] = pollmax / npts
        st2, en2 = truepart[jbest]
        recallI[ipart] = pollmax / float(en2 - st2)
    FMeasureI = {ipart: 0.0 if (precisionI[ipart] + recallI[ipart] == 0.0) else
                    2 * precisionI[ipart] * recallI[ipart] /
                    (precisionI[ipart] + recallI[ipart])
                        for ipart in precisionI}
    FMeasure = 0.0
    nparts = len(FMeasureI)
    for ipart, val in FMeasureI.iteritems():
        FMeasure += val
    FMeasure /= nparts
    return FMeasure, FMeasureI


def ConditionalEntropy(testpart, truepart, probnorm='full'):
    """
    Calculate conditional entropy of true given test.
    Parameter probnorm in the set {'full', 'test', 'true'}.
    """
    testtruepoll = _get_testtruepoll(testpart, truepart)
    nval = float(_get_probabilityNorm(testpart, truepart, probnorm))
    nptsI = {}
    for ipart, (st, en) in testpart.iteritems():
        nptsI[ipart] = float(en - st)
    HTrueGivenTestI = {}
    for ipart, (st, en) in testpart.iteritems():
        ni = nptsI[ipart]
        sumval = 0.0
        for jpart in truepart:
            nij = float(testtruepoll[ipart][jpart])
            if nij == 0.0:
                continue
            sumval -= nij / ni * np.log(nij / ni)
        HTrueGivenTestI[ipart] = sumval
    HTrueGivenTest = np.sum([HTrueGivenTestI[ipart] * nptsI[ipart] / nval
                            for ipart in testpart])
    return HTrueGivenTest, HTrueGivenTestI


def NormalizedMI(testpart, truepart, probnorm='full'):
    """
    Calculate normalized mutual information between true and test.
    Parameter probnorm in the set {'full', 'test', 'true'}.
    """
    testtruepoll = _get_testtruepoll(testpart, truepart)
    nval = float(_get_probabilityNorm(testpart, truepart, probnorm))
    nptsI = {}
    for ipart, (st, en) in testpart.iteritems():
        nptsI[ipart] = float(en - st)
    nptsJ = {}
    for jpart, (st, en) in truepart.iteritems():
        nptsJ[jpart] = float(en - st)
    HTrue = 0.0
    for jpart in truepart:
        npts = nptsJ[jpart]
        if npts == 0.0:
            continue
        HTrue -= npts / nval * np.log(npts / nval)
    HTest = 0.0
    for ipart in testpart:
        npts = nptsI[ipart]
        if npts == 0.0:
            continue
        HTest -= npts / nval * np.log(npts / nval)
    MI = 0.0
    for ipart in testpart:
        for jpart in truepart:
            nij = float(testtruepoll[ipart][jpart])
            if nij == 0.0:
                continue
            MI += nij / nval * np.log((nij * nval) /
                        (nptsI[ipart] * nptsJ[jpart]))
    NMI = MI / np.sqrt(HTest * HTrue)
    return NMI


def Jaccard(testpart, truepart):
    """
    Get Jaccard coefficient.
    """
    TP, FP, FN = _get_TF_PN(testpart, truepart)
    return TP / float(TP + FN + FP)


def FowlkesMallow(testpart, truepart):
    """
    Get Fowlkes-Mallow measure.
    """
    TP, FP, FN = _get_TF_PN(testpart, truepart)
    return TP / np.sqrt((TP + FN) * (TP + FP))


def fullAnalysis(testpart, truepart, probnorm='full',
                 output=True, retvals=True, printDetails=False):
    """
    Perform analysis on all measures, with option to print output.
    """
    testtruepoll = _get_testtruepoll(testpart, truepart)
    p, pI = purity(testpart, truepart)
    FMeas, FMeasI = FMeasure(testpart, truepart)
    HTrueGivenTest, HTrueGivenTestI = ConditionalEntropy(testpart, truepart,
                    probnorm=probnorm)
    NMI = NormalizedMI(testpart, truepart, probnorm=probnorm)
    J = Jaccard(testpart, truepart)
    FM = FowlkesMallow(testpart, truepart)
    if output:
        print
        print '----------------------------'
        print 'Matching Analysis:'
        print 'Purity =', p, ((',', pI) if printDetails else '')
        print 'F-Measure =', FMeas, ((',', FMeasI) if printDetails else '')
        print '----------------------------'
        print 'Entropy Analysis:'
        print 'Conditional entropy =', HTrueGivenTest, ((',', HTrueGivenTestI)
                        if printDetails else '')
        print 'Normalized MI =', NMI
        print '----------------------------'
        print 'Pairwise Analysis:'
        print 'Jaccard coefficient =', J
        print 'Fowlkes-Mallow measure =', FM
        print
    if retvals:
        return {'purity': (p, pI), 'F-Measure': (FMeas, FMeasI),
                'Conditional Entropy': (HTrueGivenTest, HTrueGivenTestI),
                'Normalized MI': NMI, 'Jaccard': J, 'Fowlkes-Mallard': FM}


#############################

# Comparing partition boundaries with epigenetic extrema


def _get_epigenTrackVsPartition(plimuniq, track, res, compare='min',
                sigma=10.0, zoomfactor=0.25):
    """
    Get data to compare extrema of epigenetic tracks with partition boundaries.
    rReturns: array data...
              data[0] = partition bounds
              data[1] = nearest extremum
              data[2] = distance between closest extremum pair
    """
    func = np.greater_equal if (compare == 'max') else np.less_equal
    factor = res / zoomfactor / 1.0e6
    zoomedtrack = zoom(gaussian_filter(track, sigma, mode='nearest'),
                       zoomfactor)
    zeroval = np.max(zoomedtrack) * 1.0e-6
    extrema = list(argrelextrema(zoomedtrack, func)[0])
    if compare == 'min':
        # Include zeros
        extrema.extend(list(np.nonzero(zoomedtrack < zeroval)[0]))
    extrema = np.sort(np.unique(extrema))
    data = np.zeros((3, len(plimuniq) + 2))
    en = len(zoomedtrack) * factor
    data[:, -1] = en, en, 0.0
    for i, plim in enumerate(plimuniq):
        plimz = plim * zoomfactor
        #nearestmin = extrema[np.argmin(np.abs(plimz - extrema))]
        trial = np.nonzero(extrema > plimz)[0]
        if len(trial) == 0:
            nearestmin = extrema[-1] if ((len(zoomedtrack) - plimz) >
                    (plimz - extrema[-1])) else len(zoomedtrack)
            ind = -1
            err = (len(zoomedtrack) - extrema[ind]) / 2.0
        else:
            nearestmin = extrema[np.argmin(np.abs(plimz - extrema))]
            ind = np.min(trial)
            if ind == 0:
                err = extrema[ind] / 2.0
            else:
                err = (extrema[ind] - extrema[ind - 1]) / 2.0
        data[:, i + 1] = (plimz * factor, nearestmin * factor, err * factor)
    return data


def _plot_epigenTrackVsPartition(data, ax=None, colors=None):
    """
    Plot epigen track extrema vs partition boundaries.
    """
    if ax is None:
        retfigax = True
        f, ax = plt.subplots(1, 1, figsize=(12, 12))
    else:
        retfigax = False
    if colors is None:
        colors = ['r', 'b', 'g']
    _ = ax.scatter(data[0, 1:-1], data[1, 1:-1], marker='o', c=colors[1])
    _ = ax.plot([0, np.max(data[:2])], [0, np.max(data[:2])], '--', c=colors[0])
    #_ = ax.fill_between(data[0], data[0] - data[2], data[0] + data[2],
                    #facecolor=colors[2], alpha=0.5, lw=0.0)
    _ = ax.set_xlim(0, np.max(data[:2]))
    _ = ax.set_ylim(0, np.max(data[:2]))
    _ = ax.set_xlabel('Partition boundaries')
    _ = ax.set_ylabel('Epigenetic mark extrema')
    if retfigax:
        return f, ax


def _plot_epigenTrackVsPartition_flat(data, ax=None, maxerr=None, colors=None):
    """
    Plot epigen track extrema vs partition boundaries.
    """
    if ax is None:
        retfigax = True
        f, ax = plt.subplots(1, 1, figsize=(12, 12))
    else:
        retfigax = False
    if colors is None:
        colors = ['r', 'b', 'g']
    _ = ax.scatter(data[0, 1:-1], (data[1, 1:-1] - data[0, 1:-1]) / data[2, 1:-1],
                    marker='x', c=colors[1])
    _ = ax.plot([0, np.max(data[:2])], [0, 0], '--', c=colors[0])
    #_ = ax.fill_between(data[0], -data[2], data[2],
                    #facecolor=colors[2], alpha=0.5, lw=0.0)
    _ = ax.set_xlim(0, np.max(data[:2]))
    if maxerr is None:
        maxerr = np.max(data[2])
    _ = ax.set_ylim(-maxerr, maxerr)
    _ = ax.set_xlabel('Partition boundaries')
    _ = ax.set_ylabel('Distance from epigenetic extrema')
    if retfigax:
        return f, ax


def _plot_epigenTrackVsPartition_flatAbs(data, ax=None, colors=None,
                maxerr=None):
    """
    Plot epigen track extrema vs partition boundaries.
    """
    if ax is None:
        retfigax = True
        f, ax = plt.subplots(1, 1, figsize=(12, 12))
    else:
        retfigax = False
    if colors is None:
        colors = ['r', 'b', 'g']
    if maxerr is None:
        maxerr = 1.0
    _ = ax.scatter(data[0, 1:-1], np.abs(data[1, 1:-1] - data[0, 1:-1]) /
                    data[2, 1:-1] * maxerr,
                    marker='x', c=colors[1])
    #_ = ax.plot([0, np.max(data[:2])], [0, 0], '--', c=colors[0])
    #_ = ax.fill_between(data[0], -data[2], data[2],
                    #facecolor=colors[2], alpha=0.5, lw=0.0)
    _ = ax.set_xlim(0, np.max(data[:2]))
    _ = ax.set_ylim(0, maxerr)
    #_ = ax.set_xlabel('Partition boundaries')
    #_ = ax.set_ylabel('Distance from epigenetic extrema')
    if retfigax:
        return f, ax


#############################

if __name__ == '__main__':
    truepart = {0: [0, 20], 1: [25, 40], 2: [40, 60]}
    testpart = {0: [0, 20], 1: [25, 39], 2: [39, 50], 3: [50, 60]}
    probnorm = 'full'
    testtruepoll = _get_testtruepoll(testpart, truepart)
    print
    print 'partitioncompare: Module for partitioning comparison metrics.'
    print
    print '----------------------------'
    print 'Sample partitionings:'
    print 'True:', truepart
    print 'Test:', testpart
    print 'Poll:'
    _print_testtruepoll(testtruepoll)
    fullAnalysis(testpart, truepart, probnorm=probnorm,
                 output=True, retvals=False)
