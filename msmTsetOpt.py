#!/usr/bin/env python
"""
Collection of target set optimization functions and utilities for Hi-C analysis

Part of ChromaWalker package
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
from operator import itemgetter


# Target set optimization


def _trynewtarget_construct(cmat, targetset, rhofunc=mt._rhoindex):
    nbins = len(cmat)
    tset = list(targetset)
    scores = []
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
                allchoices, nstep, pstep, kT, minstep=1, rhofunc=mt._rhoindex):
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
                #print 'Gamma move', j, 'good'
                bestrho = trialrho
                besttset = copy.deepcopy(trialset)
    return targetset, bestrho, besttset


def _stepping_random(targetset, cmat, bestrho, besttset,
                allchoices, nstep, pstep, kT, minstep=1, rhofunc=mt._rhoindex):
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
                #print 'Random move', j, 'good'
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


def _ConstructTset_optimize2(cmat, rhofunc=mt._rhoindex):
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
    dataset = dfr._get_tsetdataset2(pars)
    chrfullname = pars['cname']
    region = pars['region']
    tsetdatadir = pars['tsetdatadir']
    tsetdataprefix = pars['tsetdataprefix']
    rhomodesfx = mt._get_rhomodesfx(rhomode)
    rhodict0, tsetdict0 = dfr._get_rhotsetdicts_20160801(tsetdatadir,
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
            hfio._pickle_securedump(rfname, rhodict0, freed=True)
    if key in rhodict0:
        print 'Load 2-target state from data'
        bestrho = rhodict0[key]
        targetset = tsetdict0[key]
    else:
        print 'Seed 2-target state from pivec'
        fmat, _, cmat, _ = dfr._get_arrays(dfr._get_runbinarydir(pars),
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
                gammamaxmap, rhofunc=mt._rhoindex):
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
    Method explained in Chromatin manuscript: basically using "guided" Monte
        Carlo and "random" Monte Carlo in succession, with
        parameters defined by steppars.
    """
    res = pars['res']
    beta = pars['beta']
    norm = pars['norm']
    nstep_gammamax, pstep_gammamax, nstep_random, pstep_random, kT = steppars
    rhodata = {}
    tsetdata = {}
    # Rho mode
    rhofunc = mt._get_rhofunc(rhomode)
    # Read cmat, fmat, mmat, mapping
    fmat, mmat, cmat, (mapping, _) = dfr._get_arrays(
                        dfr._get_runbinarydir(pars), norm=norm)
    nbins = len(cmat)
    allchoices = range(nbins)
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
    #############################
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
    #############################
    ## For each new target:
    #print type(targetset), targetset
    for i in range(2, ntargetmax):
        ntarget = i + 1
        st = time()
        ### Optimize new target
        newtarget, newrho = _trynewtarget_construct(cmat, targetset,
                        rhofunc=rhofunc)
        targetset.append(newtarget)
        if len(targetset) != ntarget:
            print 'Error! ntarget mismatch 1!'
            sys.exit()
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


def run_PertSnap_fullarray(pars, pertpars, meansize=1.333, initmode=False,
                exhaustive=False, rhomode='frac'):
    """
    Run PertSnap optimization for rho on a range of ntarget: [2, ntargetmax]
    Set ntargetmax / mappedlength ~ 0.75 Mb^-1
    Fine-tuning step in addition to what's explained in Chromatin manuscript.
        Basically, randomly select a single target to perturb left/right
        along the chromosome to see if rho improves.
    For each selected target, we try positions within +-nstep bins
        of the original. Then perform this random selection ntrials times.
    """
    res = pars['res']
    beta = pars['beta']
    norm = pars['norm']
    tsetdatadir = pars['tsetdatadir']
    tsetdataprefix = pars['tsetdataprefix']
    chrfullname = pars['cname']
    region = pars.get('region', 'full')
    rhomodesfx = mt._get_rhomodesfx(rhomode)
    ntrials, nstep = pertpars
    newrhodict = {}
    newtsetdict = {}
    rhodict, tsetdict = hcu._get_rhotsetdicts_20160801(tsetdatadir,
            tsetdataprefix, chrfullname, region, res,
            hcu._get_tsetdataset2(pars), rhomodesfx=rhomodesfx)
    # Rho mode
    rhofunc = mt._get_rhofunc(rhomode)
    # Read cmat, fmat, mmat, mapping
    fmat, mmat, cmat, (mapping, _) = dfr._get_arrays(
                        dfr._get_runbinarydir(pars), norm=norm)
    nbins = len(cmat)
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
    for ntarget in range(2, ntargetmax):
        print 'ntarget', ntarget, '...',
        st = time()
        key = (beta, ntarget)
        if key not in rhodict:
            print 'break!'
            break
        if key in rhodict and key not in tsetdict:
            print 'break: bad tset!'
            break
        bestrho = rhodict[key]
        initrho = bestrho
        tset = tsetdict[key]
        moved = False
        for i in range(ntrials):
            testind = np.random.randint(0, ntarget)
            testtset0 = np.array(tset).copy()
            for step in range(1, nstep + 1):
                testtset = testtset0.copy()
                testtset[testind] = max(0, testtset[testind] - step)
                if len(np.unique(testtset)) < ntarget:
                    continue
                testrho = rhofunc(cmat, testtset)
                if testrho < bestrho:
                    rho12 = testrho, initrho
                    moved = True
                    tset = testtset.copy()
                    bestrho = testrho
        for i in range(ntrials):
            testind = np.random.randint(0, ntarget)
            testtset0 = np.array(tset).copy()
            for step in range(1, nstep + 1):
                testtset = testtset0.copy()
                testtset[testind] = min(ntarget - 1,
                                testtset[testind] + step)
                if len(np.unique(testtset)) < ntarget:
                    continue
                testrho = rhofunc(cmat, testtset)
                if testrho < bestrho:
                    rho12 = testrho, initrho
                    moved = True
                    tset = testtset.copy()
                    bestrho = testrho
        en = time()
        print (': %.3e secs' % (en - st)),
        if moved:
            print 'moved! %.3e' % bestrho
            newtsetdict[key] = tset
            newrhodict[key] = bestrho
        else:
            print
    return newrhodict, newtsetdict


# Merge old and new rho/tset dicts


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


# Automatically finding optimal levels


def _find_goodLevels(rholist, ntargetlist, mappedchrlen, rhomax=0.8,
                skipfirst=None):
    """
    Define optimal levels of structural hierarchy (minima in rho) below rhomax.
    Returns list of ntarget values.
    If skipfirst is True, skip first minimum after ntarget = 2.
    If None, set skipfirst to True if mappedchrlen > 200Mbp.
    """
    if skipfirst is None:
        skipfirst = (mappedchrlen > 2e8)
    rhoarr = np.array(rholist)
    ntarr = np.array(ntargetlist)
    # Find minima: Ignore first and last ntarget values
    nextgreater = rhoarr[1:] > rhoarr[:-1]
    prevgreater = rhoarr[:-1] > rhoarr[1:]
    minmask = np.array(nextgreater[1:] * prevgreater[:-1], dtype=bool)
    minmask = np.array(minmask * (rhoarr[1:-1] < rhomax), dtype=bool)
    ntmin = ntarr[1:-1][minmask]
    if skipfirst:
        ntmin = ntmin[1:]
    return np.array(ntmin)


def _find_optimalLevels(rholist, ntargetlist, mappedchrlen, rhomax=0.8,
                skipfirst=None):
    """
    Define optimal levels of structural hierarchy (minima in rho) below rhomax.
    Returns list of ntarget values.
    If skipfirst is True, skip first minimum after ntarget = 2.
    If None, set skipfirst to True if mappedchrlen > 200Mbp.
    """
    #if skipfirst is None:
        #skipfirst = (mappedchrlen > 2e8)
    #rhoarr = np.array(rholist)
    #ntarr = np.array(ntargetlist)
    ## Find minima: Ignore first and last ntarget values
    #nextgreater = rhoarr[1:] > rhoarr[:-1]
    #prevgreater = rhoarr[:-1] > rhoarr[1:]
    #minmask = np.array(nextgreater[1:] * prevgreater[:-1], dtype=bool)
    #minmask = np.array(minmask * (rhoarr[1:-1] < rhomax), dtype=bool)
    #ntmin = ntarr[1:-1][minmask]
    #if skipfirst:
        #ntmin = ntmin[1:]
    ntmin = _find_goodLevels(rholist, ntargetlist, mappedchrlen, rhomax=rhomax,
                skipfirst=skipfirst)
    # Set ntmin[i+1] > 2*ntmin[i]
    ntmin = list(ntmin)
    print ntmin
    i = 0
    while True:
        thismin = ntmin[i]
        while True:
            if len(ntmin) == i + 1:
                break
            nextmin = ntmin[i + 1]
            if nextmin < 2 * thismin:
                ntmin.pop(ntmin.index(nextmin))
            else:
                i += 1
                break
        if i + 1 >= len(ntmin):
            break
    return np.array(ntmin)


################################################

# To deprecate...


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
        bestrho = mt._rhoindex(cmat, targetset)
        for i in range(nbins):
            for j in range(i + 1, nbins):
                trialset = [i, j]
                thisrho = mt._rhoindex(cmat, trialset)
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
        rhodict0, tsetdict0 = hcu._get_rhotsetdicts(tsetdatadir,
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
    # Rho mode
    rhofunc = mt._get_rhofunc(rhomode)
    # Read cmat, fmat, mmat, mapping
    nbins = len(cmat)
    allchoices = range(nbins)
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
            sys.exit()
        key = (beta, 2)
        rhodata[key] = bestrho
        tsetdata[key] = copy.deepcopy(targetset)
    else:
        st = time()
        newrho, targetset = _ConstructTset_dictpivec2_indict(
                            beta, arrays, indicts)
        if len(targetset) != 2:
            sys.exit()
        bestrho, besttset = _ConstructTset_MCn(steppars, newrho,
                        targetset, allchoices, cmat, gammamaxmap,
                        rhofunc=rhofunc)
        targetset = list(besttset)
        if len(targetset) != 2:
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
            sys.exit()
        #print targetset, 'c'
        bestrho, besttset = _ConstructTset_MCn(steppars, newrho,
                        targetset, allchoices, cmat, gammamaxmap,
                        rhofunc=rhofunc)
        ### Record rhodata, targetsetdata
        targetset = besttset
        if len(targetset) != ntarget:
            sys.exit()
        key = (beta, ntarget)
        rhodata[key] = bestrho
        tsetdata[key] = copy.deepcopy(targetset)
        en = time()
        print i + 1, ('(%.2e secs):' % (en - st)), bestrho
        #print targetset
    return rhodata, tsetdata


# Consensus target set determination using equal size k-means clustering


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


#################################################################

# TargetsetOptimizer class:
#    - Run ConstructMC / PertSnap routines
#    - TODO: Comparison across beta values, consensus polling

class TargetOptimizer:
    """
    Container to initialize / hold parameters pertaining to targetset
    optimization in HiC analysis runs and simplify function calls.
    TODO: Add consensus polling for defining reference tsets.
    """
    def __init__(self, pars, DFR=None, conMCpars=None, pertpars=None):
        """
        initialize TargetOptimizer instance.
        """
        if DFR is None:
            self.DFR= dfr.DataFileReader(pars)
        else:
            self.DFR = DFR

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
        ## TPT optimization depth: Smallest average partition size in Mbp?
        self.meansize = pars.get('meansize', 0.8)
        ## TPT partition definition: Partitions smaller than this size should
        ##                           be merged with neighbors (in Mbp)
        self.minpartsize = pars.get('minpartsize', 0.5)
        ## TPT partition definition: Criterion for metastability index rho
        self.rhomax = pars.get('rhomax', 0.8)
        ## TPT partition definition: Skip first minimum in rho?
        self.skipfirst = pars.get('skipfirst', None)
        ## TPT Laplacian definition: Thermal annealing beta of interaction
        ##                matrix used for Laplacian computation
        self.fmatbeta = pars.get('fmatbeta', 1.0)
        ## TPT Laplacian definition: For binning mode, normalize matrix by
        ##                partition sizes?
        self.matnorm = pars.get('matnorm', 'sum')

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

        ##############################
        # ConMC / pSnap parameters
        if conMCpars is None:
            self.steppars = 100, 0.5, 500, 0.1, 1.0
        else:
            self.steppars = (conMCpars.get('nstep_gammamax', 100),
                             conMCpars.get('pstep_gammamax', 0.5),
                             conMCpars.get('nstep_random', 200),
                             conMCpars.get('pstep_random', 0.1),
                             conMCpars.get('kT', 1.0))
        if pertpars is None:
            self.pertpars = 200, 4
        else:
            self.pertpars = (pertpars.get('ntrials', 200),
                             pertpars.get('nstep', 4))

    def seed(self, cname, beta, exhaustive=None):
        """
        Find seed 2-target set for given case.
        Parameter 'exhaustive' can be True/False/None.
            If None, will perform exhaustive target-pair search only if matrix
            size is no more than N = 1500.
        """
        if exhaustive is None:
            size = len(self.DFR.get_mappingdata(cname, beta)[0])
            exhaustive = (size <= 1500)
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname
        thispar['beta'] = beta
        rd, td = run_ConstructMC_fullarray(thispar, self.steppars,
                            meansize=self.meansize, initmode=True,
                            exhaustive=exhaustive, rhomode=self.rhomode)
        self.DFR.update_datadicts(cname, rd, td)

    def conMC(self, cname, beta):
        """
        Perform targetset search / optimization using ConstructMC routine.
        """
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname
        thispar['beta'] = beta
        rd, td = run_ConstructMC_fullarray(thispar, self.steppars,
                            meansize=self.meansize, rhomode=self.rhomode)
        return self.DFR.update_datadicts(cname, rd, td)

    def pSnap(self, cname, beta):
        """
        Perform targetset search / optimization using PertSnap routine.
        """
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname
        thispar['beta'] = beta
        rd, td = run_PertSnap_fullarray(thispar, self.pertpars,
                            meansize=self.meansize, rhomode='frac')
        return self.DFR.update_datadicts(cname, rd, td)

    def get_partitions(self, cname, beta, ntarget):
        """
        Get partitioning information for given chromosome, with annealing beta,
        with number of targets ntarget.
        Each locus on the network at beta = fmatbeta will be assigned
        to a partition. This is to ease computation of effective interactions
        using the observed/base interaction matrix.
        """
        # Get tset
        rd, td = self.DFR.get_datadicts(cname)
        _, tset = self.DFR.readout_datadicts(rd, td, beta, ntarget)
        # Get hard, split and padded membership functions
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname
        thispar['tsetbeta'] = beta
        cutoffsize = thispar['minpartsize'] / thispar['res']
        membership = mt._get_mappedPaddedMembership(thispar, tset,
                norm=thispar['norm'], merge=True, cutoffsize=cutoffsize,
                split=True)
        # Get partition information
        lims, ids = mt._get_limsIds_fromPaddedSplitMembership(membership)
        return membership, lims, ids

    def _get_rholist(self, cname, beta, ntargetmax=None):
        """
        Get list of ntarget / rho with data available.
        """
        rd, td = self.DFR.get_datadicts(cname)
        ntlist = []
        rlist = []
        if ntargetmax is None:
            n = 2
            while True:
                data = self.DFR.readout_datadicts(rd, td, beta, n)
                if data is not None:
                    ntlist.append(n)
                    rlist.append(data[0])
                    n += 1
                else:
                    break
        else:
            for n in range(2, ntargetmax):
                data = self.DFR.readout_datadicts(rd, td, beta, n)
                if data is not None:
                    ntlist.append(n)
                    rlist.append(data[0])
        return rlist, ntlist

    def get_goodLevels(self, cname, beta, ntargetmax=None):
        """
        Get all optimal levels of hierarchy in a given chromosome, with
        thermal annealing beta, with metastability index below rhomax.
        Returns list of ntarget values.

        If skipfirst is True, skip first minimum after ntarget = 2.
        If None, set skipfirst to True if mappedchrlen > 200Mbp.
        """
        # Get list of ntarget and rho
        rlist, ntlist = self._get_rholist(cname, beta, ntargetmax=ntargetmax)
        # Find mapped length along chromosome
        mappingdata = self.DFR.get_mappingdata(cname, beta)
        mappedchrlen = len(mappingdata[0]) * self.res
        # Calculate optimal levels
        return _find_goodLevels(rlist, ntlist, mappedchrlen,
                rhomax=self.rhomax, skipfirst=self.skipfirst)

    def get_optimalLevels(self, cname, beta, ntargetmax=None):
        """
        Get all optimal levels of hierarchy in a given chromosome, with
        thermal annealing beta, with metastability index below rhomax.
        Returns list of ntarget values.

        If skipfirst is True, skip first minimum after ntarget = 2.
        If None, set skipfirst to True if mappedchrlen > 200Mbp.
        """
        # Get list of ntarget and rho
        rlist, ntlist = self._get_rholist(cname, beta, ntargetmax=ntargetmax)
        # Find mapped length along chromosome
        mappingdata = self.DFR.get_mappingdata(cname, beta)
        mappedchrlen = len(mappingdata[0]) * self.res
        # Calculate optimal levels
        return _find_optimalLevels(rlist, ntlist, mappedchrlen,
                rhomax=self.rhomax, skipfirst=self.skipfirst)

    def plot_partitionHierarchy_all(self, ax, cname, beta, ntargetmax=40):
        """
        Plot hierarchy of partitions up to ntargetmax.
        """
        # Find true max ntarget
        rd, td = self.DFR.get_datadicts(cname)
        ntlist = []
        limslist = []
        idslist = []
        clrlist = []
        for nt in range(2, ntargetmax):
            data = self.DFR.readout_datadicts(rd, td, beta, nt)
            if data is not None:
                # Get lims/ids, and define color
                ntlist.append(nt)
                _, lims, ids = self.get_partitions(cname, beta, nt)
                limslist.append(lims)
                idslist.append(ids)
                clr = plu._get_partitionlevelcolor(nt, 2, ntargetmax,
                                mode='pow', n=-1.0)
                clrlist.append(clr)
        # Plot
        plu._plot_PartitionsHierarchy(ax, limslist, idslist, clrlist, ntlist,
                [], self.res)

    def plot_partitionHierarchy_optimal(self, ax, cname, beta, ntargetmax=None,
                    rhomax=0.8, skipfirst=None, optimal=True):
        """
        Plot hierarchy of partitions up to ntargetmax.
        If optimal set to False, include all good ntarget values.
        """
        # Find true max ntarget
        rd, td = self.DFR.get_datadicts(cname)
        if optimal:
            ntlist = self.get_optimalLevels(cname, beta, ntargetmax=ntargetmax)
            optlist = []
        else:
            ntlist = self.get_goodLevels(cname, beta, ntargetmax=ntargetmax)
            optlist = self.get_optimalLevels(cname, beta, ntargetmax=ntargetmax)
        limslist = []
        idslist = []
        clrlist = []
        for nt in ntlist:
            # Get lims/ids, and define color
            _, lims, ids = self.get_partitions(cname, beta, nt)
            limslist.append(lims)
            idslist.append(ids)
            clr = plu._get_partitionlevelcolor(nt, 2, 100,
                            mode='pow', n=-1.0)
            clrlist.append(clr)
        # Plot
        plu._plot_PartitionsHierarchy(ax, limslist, idslist, clrlist, ntlist,
                optlist, self.res)

    def get_binLaplacian(self, cname, beta, ntarget):
        """
        Compute Laplacian obtained by binning operation on hard partitions,
        for intra-chromosomal network.
        """
        # Get tset
        rd, td = self.DFR.get_datadicts(cname)
        _, tset = self.DFR.readout_datadicts(rd, td, beta, ntarget)
        # Get hard, split and padded membership functions
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname
        thispar['tsetbeta'] = beta
        cutoffsize = thispar['minpartsize'] / thispar['res']
        membership = mt._get_mappedPaddedMembership(thispar, tset,
                norm=thispar['norm'], merge=True, cutoffsize=cutoffsize,
                split=True)
        # Compute Laplacian
        npartitions = len(membership)
        if self.matnorm == 'average':
            for i in range(npartitions):
                membership[i] /= np.sum(membership[i])
        elif self.matnorm == 'sqrt':
            for i in range(npartitions):
                membership[i] /= np.sqrt(np.sum(membership[i]))
        fmat = self.DFR.get_fmat(cname, self.fmatbeta)
        mappingdata = self.DFR.get_mappingdata(cname, self.fmatbeta)
        fmatpad = plu._build_fullarray(fmat, mappingdata, 0.0)
        lab = np.array([[np.dot(np.dot(membership[i], fmatpad), membership[j])
                         for j in range(npartitions)]
                        for i in range(npartitions)])
        lab2 = np.dot(membership, np.dot(fmatpad, membership.T))
        if not np.allclose(lab, lab2):
            print 'Laplacians not the same!'
        lab -= np.diag(np.diag(lab))
        return lab

    def get_binLaplacian_inter(self, cname1, cname2, beta1, beta2,
                    ntarget1, ntarget2):
        """
        Compute Laplacian obtained by binning operation on hard partitions,
        across 2 chromosomes.
        """
        ##########################
        # Get chr1 membership
        # Get tset
        rd, td = self.DFR.get_datadicts(cname1)
        _, tset = self.DFR.readout_datadicts(rd, td, beta1, ntarget1)
        # Get hard, split and padded membership functions
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname1
        thispar['tsetbeta'] = beta1
        cutoffsize = thispar['minpartsize'] / thispar['res']
        membership1 = mt._get_mappedPaddedMembership(thispar, tset,
                norm=thispar['norm'], merge=True, cutoffsize=cutoffsize,
                split=True)
        ##########################
        # Get chr2 membership
        # Get tset
        rd, td = self.DFR.get_datadicts(cname2)
        _, tset = self.DFR.readout_datadicts(rd, td, beta2, ntarget2)
        # Get hard, split and padded membership functions
        thispar = copy.deepcopy(self.basepars)
        thispar['cname'] = cname2
        thispar['tsetbeta'] = beta2
        cutoffsize = thispar['minpartsize'] / thispar['res']
        membership2 = mt._get_mappedPaddedMembership(thispar, tset,
                norm=thispar['norm'], merge=True, cutoffsize=cutoffsize,
                split=True)
        ##########################
        # Get padded inter-chr fmat
        fmat, md1, md2 = self.DFR.get_fmatMapdata_inter(cname1, cname2)
        ## Appli beta to fmat... (We use fmatbeta = 1.0 by default)
        fmat = fmat ** self.fmatbeta
        ##########################
        # Get binned Laplacian
        fmatpad = plu._build_fullarray_inter(fmat, md1, md2, 0.0)
        lab = np.dot(membership1, np.dot(fmatpad, membership2.T))
        return lab


if __name__ == '__main__':
    ################################
    # Perform tests on class functions
    plt.ion()
    print
    print '**********************************************'
    print 'Welcome to msmTsetOpt test suite!'
    print '**********************************************'
    print
    baseres = 50000
    res = 50000
    cname = 'chr22'
    cname2 = 'chr21'
    beta = 7.0
    norm='gfilter_2e5'
    meansize = 2.0
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
            'res': res,
            'norm': norm,
            'meansize': meansize,
            }
    epigenpars = {'epigendatadir': 'epigenomic-tracks'}
    print 'Create DFR object...'
    DFR = dfr.DataFileReader(pars, epigenpars=epigenpars)
    # Load and compute f/m/cmat
    print 'Compute f/m/cmat...'
    fmat, mmat, cmat, mappingdata = DFR.get_arrays(cname2, 1.0)
    fmat, mmat, cmat, mappingdata = DFR.get_arrays(cname2, beta)
    fmat, mmat, cmat, mappingdata = DFR.get_arrays(cname, 1.0)
    fmat, mmat, cmat, mappingdata = DFR.get_arrays(cname, beta)
    TOpt = TargetOptimizer(pars, DFR=DFR)
    # Plot matrices
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
    _ = raw_input('Enter anything to continue:')
    plt.close('all')
    print
    # Seed 2-target set
    print 'Seeding...'
    TOpt.seed(cname, beta, exhaustive=False)
    TOpt.seed(cname2, beta, exhaustive=False)
    # ConstructMC
    print 'ConstructMC...'
    TOpt.conMC(cname, beta)
    TOpt.conMC(cname2, beta)
    # PertSnap
    print 'PertSnap...'
    TOpt.pSnap(cname, beta)
    TOpt.pSnap(cname2, beta)
    ######################################
    # Display optimization results...
    print
    print '**********************************************'
    print
    f, x = plt.subplots(1, 1, figsize=(6, 4))
    rd, td = DFR.get_datadicts(cname)
    rho = []
    n = []
    for i in range(2, 12):
        rval, tset = DFR.readout_datadicts(rd, td, beta, i)
        rho.append(min(rval, 1.0))
        n.append(i)
    _ = x.plot(n, rho, label=cname)
    rd, td = DFR.get_datadicts(cname2)
    rho = []
    n = []
    for i in range(2, 12):
        rval, tset = DFR.readout_datadicts(rd, td, beta, i)
        rho.append(min(rval, 1.0))
        n.append(i)
    _ = x.plot(n, rho, label=cname2)
    _ = x.legend()
    _ = x.set_title('%s metastability index trace $\\rho_n$' % cname)
    _ = raw_input('Enter anything to continue:')
    plt.close('all')
    ######################################
    # Partition hierarchy...
    f, x = plt.subplots(1, 1, figsize=(6, 10))
    TOpt.plot_partitionHierarchy_all(x, cname, beta)
    x.set_title('%s: All levels' % cname)
    f, x = plt.subplots(1, 1, figsize=(6, 3))
    TOpt.plot_partitionHierarchy_optimal(x, cname, beta)
    x.set_title('%s: Optimal levels' % cname)
    _ = raw_input('Enter anything to continue:')
    plt.close('all')
    f, x = plt.subplots(1, 1, figsize=(6, 10))
    TOpt.plot_partitionHierarchy_all(x, cname2, beta)
    x.set_title('%s: All levels' % cname2)
    f, x = plt.subplots(1, 1, figsize=(6, 3))
    TOpt.plot_partitionHierarchy_optimal(x, cname2, beta)
    x.set_title('%s: Optimal levels' % cname2)
    _ = raw_input('Enter anything to continue:')
    plt.close('all')
    ######################################
    # Effective interactions...
    lab = TOpt.get_binLaplacian(cname, beta, 4)
    f, x = plt.subplots()
    x.matshow(lab)
    lab = TOpt.get_binLaplacian_inter(cname2, cname, beta, beta, 10, 10)
    f, x = plt.subplots()
    x.matshow(lab)
    _ = raw_input('Enter anything to exit:')
    print 'Adios!'
    print

