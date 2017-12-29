#!/usr/bin/env python
"""
Collection of TPT-related functions and utilities for Hi-C analysis

Part of ChromaWalker package
"""

import numpy as np
import copy
from numpy.linalg import solve
import plotutils as plu
import hicFileIO as hfio
import dataFileReader as dfr
import msmBasics as mb
import msmBookKeeping as mbk


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


def _rhoindex(cmat, targetset, rhomode='frac'):
    """
    Compute metastability index for given target set.
    rhomode may be 'frac', 'num', 'denom', or 'means'.
    """
    # Calculates the metastability index from prob cmat and target set
    ## Generally...
    ##    probAB ~ probability for transitioning between targets
    ##    probiM ~ probability for transitioning into target set
    nbins = len(cmat)
    tcomp = list(set(range(nbins)) - set(targetset))
    tset = cmat[:, targetset]
    if rhomode == 'frac':
        probAB = min([1, np.max(tset[targetset])])
        probiM = min([1, min(np.max(tset[tcomp], axis=1))])
        rhoM = probAB / probiM
    elif rhomode == 'num':
        rhoM = min([1, np.max(tset[targetset])])
    elif rhomode == 'denom':
        rhoM = 1.0 / min([1, min(np.max(tset[tcomp], axis=1))])
    elif rhomode == 'means':
        probAB = min([1, np.mean(np.max(tset[targetset], axis=1))])
        probiM = min([1, np.mean(np.max(tset[tcomp], axis=1))])
        rhoM = probAB / probiM
    return rhoM


def _rhoindex_num(cmat, targetset):
    """
    Compute metastability index for given target set.
    """
    # Calculates the metastability index from prob cmat and target set
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
    tsetc = np.sort(list(set(np.arange(len(fmat))) - set(targetset)))
    qAi = {}
    for i, t in enumerate(targetset):
        thisqAi = np.zeros(len(fmat))
        thisqAi[t] = 1.0
        thisqAi[tsetc] = solve(lmat[tsetc][:, tsetc], -lmat[tsetc, t])
        qAi[t] = thisqAi
    return qAi


def _calc_qAi_sort_exact(datadir, targetset, norm='raw'):
    """
    Compute committor probabilities q_A(i), as an arrays with rows
    corresponding to targets in sorted (increasing) order.
    """
    fmat, _, _, _ = dfr._get_arrays(datadir, norm=norm)
    qAi = _calc_qAi_exact(fmat, targetset)
    tset = np.sort(targetset)
    return np.array([qAi[t] for t in tset])






###############################################

# Effective interaction computations


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
    datadir = dfr._get_runbinarydir(tpars)
    fmat, _, _, mappingdata = dfr._get_arrays(datadir, norm=norm)
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
    datadir = dfr._get_runbinarydir(tpars)
    qAi = dfr._get_TargetCommittor(datadir, tset, norm=norm)
    mappingdata = dfr._get_mappingdata(datadir, norm=norm)
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
    lab = np.dot(qAi2, lmat2)
    return lab


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
    mappedtset1 = mbk._get_tsetReference_20180802(tpars, ntarget1)[0]
    tpars['chrref'] = chrfullname1
    mapping, _ = dfr._get_mappingdata(dfr._get_runbinarydir(tpars), norm=norm)
    tset1 = np.array([list(mapping).index(t) for t in mappedtset1])
    qAc1 = dfr._get_TargetCommittor(dfr._get_runbinarydir(tpars), tset1,
                    norm=norm)
    tpars['chrfullname'] = chrfullname2
    mappedtset2 = mbk._get_tsetReference_20180802(tpars, ntarget2)[0]
    tpars['chrref'] = chrfullname2
    mapping, _ = dfr._get_mappingdata(dfr._get_runbinarydir(tpars), norm=norm)
    tset2 = np.array([list(mapping).index(t) for t in mappedtset2])
    qBc2 = dfr._get_TargetCommittor(dfr._get_runbinarydir(tpars), tset2,
                    norm=norm)
    # Get trans-fij, targetmembership
    tpars['chrref'] = chrfullname1
    fmat1, _, _, md1 = dfr._get_arrays(dfr._get_runbinarydir(tpars), norm=norm)
    tpars['chrref'] = chrfullname2
    fmat2, _, _, md2 = dfr._get_arrays(dfr._get_runbinarydir(tpars), norm=norm)
    pars['chrref'] = '_'.join([chrfullname1, chrfullname2])
    fmat12, mdp1, mdp2 = dfr._get_fmatmap_inter(pars)
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
        v1 = targetmembership[a1] * pivec
        lac1bc2[a1] = np.dot(v1, lmatcombined[:, mappedtset2 + nbins1])
    return lac1bc2


################################################

## Partitioning smoothing


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
        # 1) If embedded in single partition...
        elif lo_id == hi_id:
            this_id = lo_id
        # Compare priority
        elif lo_priority > hi_priority:
            this_id = lo_id
        elif lo_priority < hi_priority:
            this_id = hi_id
        else:
            # Find closer target
            lo_mt = mapping[tset[lo_id]]
            hi_mt = mapping[tset[hi_id]]
            this_id = lo_id if np.abs(lo - lo_mt) < np.abs(hi_mt - hi) \
                            else hi_id
        qAip2[lo:hi] = this_id
    return qAip2


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
    tsetdatadir = dfr._get_runbinarydir(tpars)
    tsetmapping, _ = dfr._get_mappingdata(dfr._get_runbinarydir(tpars), norm=norm)
    tpars['beta'] = fmatbeta
    fmatdatadir = dfr._get_runbinarydir(tpars)
    fmatmapping, _ = dfr._get_mappingdata(dfr._get_runbinarydir(tpars), norm=norm)
    tset2fmatmapping = np.array([list(fmatmapping).index(t)
                    for t in tsetmapping])
    fmat, _, _, mappingdata = dfr._get_arrays(fmatdatadir, norm=norm)
    mapping = mappingdata[0]
    fmat -= np.diag(np.diag(fmat))
    tset = np.sort(np.array(tset))
    qAi = dfr._get_TargetCommittor(tsetdatadir, tset, norm=norm)
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


################################################

## To deprecate...


#def _calc_qAi(mmat, targetset):
    #"""
    #Compute attractor membership function q_A(i).
    #"""
    #ntarget = len(targetset)
    ## qAB(i)
    #qABi = {}
    #for target in targetset:
        #complement = set(targetset) - set([target])
        #for source in complement:
            #qABi[source, target] = _calc_qijk(mmat, source, target)
    ## c_j
    #coeff = np.zeros((ntarget, ntarget - 1))
    #st = time()
    #for source in targetset:
        #qABivalues = np.ones((ntarget - 1, ntarget - 1))
        #setDiff = list(set(targetset) - set([source]))
        #for tar1 in setDiff:
            #for tar2 in setDiff:
                #qABivalues[setDiff.index(tar1), setDiff.index(tar2)] = \
                                #qABi[source, tar2][tar1]
        #coeff[list(targetset).index(source)] = \
                        #solve(qABivalues, np.ones(ntarget - 1))
    ## qA(i)
    #qAi = {i: 0 for i in targetset}
    #for source in targetset:
        #setDiff = list(set(targetset) - set([source]))
        #for target in setDiff:
            #qAi[source] += (qABi[source, target]) * \
                           #coeff[list(targetset).index(source),
                               #setDiff.index(target)]
        ## source is target and vivecersa in the committor (1-qABi)
        #qAi[source] = 1 - qAi[source]
    #return qAi


#def _calc_qAi_sort(datadir, targetset, norm='raw'):
    #"""
    #Compute committor probabilities q_A(i), as an arrays with rows
    #corresponding to targets in sorted (increasing) order.
    #"""
    #_, mmat, _, _ = dfr._get_arrays(datadir, norm=norm)
    #qAi = _calc_qAi(mmat, targetset)
    #tset = np.sort(targetset)
    #return np.array([qAi[t] for t in tset])


#def _calc_TargetEffLaplacian(datadir, tset, norm='raw'):
    #"""
    #Compute effective Laplacian between targets.
    #"""
    #fmat, mmat, cmat, mappingdata = dfr._get_arrays(datadir, norm=norm)
    #mapping = mappingdata[0]
    #tset = np.sort(np.array(tset))
    #ntarget = len(tset)
    ## Create transition matrix pmat
    #fmat -= np.diag(np.diag(fmat))
    #pivec = np.sum(fmat, axis=0)
    #pmat = np.array([v / pivec[i] for i, v in enumerate(fmat)])
    #pivec /= np.sum(pivec)
    #lmat = pmat - np.diag(np.ones_like(pivec))
    #lab = np.zeros((ntarget, ntarget))
    #for aind in range(ntarget):
        #for bind in range(ntarget):
            #if aind == bind:
                #continue
            #qab = _calc_qijk(mmat, tset[aind], tset[bind])
            #fijab = np.outer(pivec * (1.0 - qab), qab) * lmat
            #fijAB = fijab - fijab.T
            #fijAB[fijAB < 0.0] = 0.0
            #lab[aind, bind] = np.sum(fijAB[tset[aind]])
    #return lab


#def _calc_TargetEffLaplacian_20160802(datadir, tset, norm='raw'):
    #"""
    #Compute effective Laplacian between targets.
    #Based on computations / formula laid out by Enrico in:
        #J Chem Phys 145, 024102 (2016).
    #Note: L_ab is the symmetric Laplacian matrix.
    #"""
    #fmat, mmat, cmat, mappingdata = dfr._get_arrays(datadir, norm=norm)
    #fmat -= np.diag(np.diag(fmat))
    #fmat /= np.sum(fmat)
    #mapping = mappingdata[0]
    #tset = np.sort(np.array(tset))
    #ntarget = len(tset)
    #pivec = np.sum(fmat, axis=0) / np.sum(fmat)
    #qAi = dfr._get_TargetCommittor(datadir, tset, norm=norm)
    #pmat = np.array([v / pivec[i] for i, v in enumerate(fmat)])
    #lmat = pmat - np.diag(np.ones_like(pivec))
    #lmat2 = lmat[:, tset]
    #qAi2 = qAi * np.array([list(pivec)] * ntarget)
    #pivec2 = np.dot(qAi, pivec)
    ## Asymmetric k_ab transition matrix
    ##lab = np.dot(qAi2, lmat2) / np.array([list(pivec2)] * ntarget).T
    #lab = np.dot(qAi2, lmat2)
    #return lab


#def _calc_TargetBinLaplacian_20160802(datadir, tset, norm='raw',
                #membership='soft', matnorm='average'):
    #"""
    #Compute effective Laplacian between targets, based on simple binning
     #operation using hitting probabilities q_alpha.
    #membership: Use 'soft' or 'hard' partitioning
    #matnorm: Use 'average' to output average interaction per pixel^2,
             #else 'sum'.
    #Note: L_ab is the symmetric Laplacian matrix.
    #"""
    #fmat, _, _, mappingdata = dfr._get_arrays(datadir, norm=norm)
    #mapping = mappingdata[0]
    #fmat -= np.diag(np.diag(fmat))
    #fmat /= np.sum(fmat)
    #tset = np.sort(np.array(tset))
    #ntarget = len(tset)
    #qAi = dfr._get_TargetCommittor(datadir, tset, norm=norm)
    #membership = qAi.copy()
    #if membership == 'hard':
        #for i in range(qAi.shape[1]):
            #membership[:, i] = 0.0
            #membership[np.argmax(qAi[:, i]), i] = 1.0
    #if matnorm == 'average':
        #for i in range(len(membership)):
            #membership[i] /= np.sum(membership[i])
    #lab1 = np.dot(membership, np.dot(fmat, membership.T))
    #lab = np.array([[np.dot(np.dot(membership[i], fmat), membership[j])
                     #for j in range(ntarget)]
                    #for i in range(ntarget)])
    #if not np.allclose(lab, lab1):
        #print 'Not correct!'
    #del fmat
    #return lab


#def _calc_TargetBinLaplacian_20160802_mixbeta(pars, tset, norm='raw',
                #membertype='soft', matnorm='average', merge=False,
                #cutoffsize=10, nmapmax=100, split=False, getmemb=False):
    #"""
    #Compute effective Laplacian between targets, based on simple binning
     #operation using hitting probabilities q_alpha.
    #membertype: Use 'soft' or 'hard' partitioning
    #matnorm: Use 'average' to output average interaction per pixel^2.
             #Use 'sqrt' to set weight = sqrt(number of pixels).
             #Otherwise, use 'sum' to simply sum up all interaction counts.
    #merge: Used only for membertype = 'hard'
           #Merge islands that are smaller than cutoffsize pixels.
    #split: Used only for membertype = 'hard'
           #Split disconnected partitions if True.
    #getmemb: Get padded membership matrix also.
    #Note: L_ab is the symmetric Laplacian matrix.
    #"""
    #tsetbeta = pars['tsetbeta']
    #fmatbeta = pars['fmatbeta']
    #tpars = copy.deepcopy(pars)
    #tpars['beta'] = tsetbeta
    #tsetdatadir = dfr._get_runbinarydir(tpars)
    #tsetmapping, _ = dfr._get_mappingdata(dfr._get_runbinarydir(tpars), norm=norm)
    #tpars['beta'] = fmatbeta
    #fmatdatadir = dfr._get_runbinarydir(tpars)
    #fmatmapping, _ = dfr._get_mappingdata(dfr._get_runbinarydir(tpars), norm=norm)
    #mappedtset = tsetmapping[tset]
    #tset2 = [list(fmatmapping).index(t) for t in mappedtset]
    #tset2fmatmapping = np.array([list(fmatmapping).index(t)
                    #for t in tsetmapping])
    #fmat, _, _, mappingdata = dfr._get_arrays(fmatdatadir, norm=norm)
    #mapping = mappingdata[0]
    #fmat -= np.diag(np.diag(fmat))
    ##fmat /= np.sum(fmat)
    #tset = np.sort(np.array(tset))
    #qAi = dfr._get_TargetCommittor(tsetdatadir, tset, norm=norm)
    #qAip = np.zeros((len(qAi), len(fmatmapping)))
    #qAip[:, tset2fmatmapping] = qAi
    #membership = qAip.copy()
    #if membertype == 'hard':
        #for i in range(qAip.shape[1]):
            #membership[:, i] = 0.0
            #membership[np.argmax(qAip[:, i]), i] = 1.0
        #if merge or split:
            ## Pad membership array
            #membership2 = np.zeros((len(membership), mappingdata[1]))
            #membership2[:, mapping] = membership
            #if merge:
                #membership2 = _tset_mergeIslands(tset, membership2, mappingdata,
                                   #nmapmax=nmapmax, cutoffsize=cutoffsize)
            #if split:
                #membership2 = _tset_splitDisjointPartitions(membership2)
            ## Unpad membership array
            #membership = membership2[:, mapping]
    #npartitions = len(membership)
    #if matnorm == 'average':
        #for i in range(len(membership)):
            #membership[i] /= np.sum(membership[i])
    #elif matnorm == 'sqrt':
        #for i in range(len(membership)):
            #membership[i] /= np.sqrt(np.sum(membership[i]))
    #lab1 = np.dot(membership, np.dot(fmat, membership.T))
    #lab = np.array([[np.dot(np.dot(membership[i], fmat), membership[j])
                     #for j in range(npartitions)]
                    #for i in range(npartitions)])
    #if not np.allclose(lab, lab1):
        #print 'Not correct!'
    #del fmat
    #if getmemb:
        #return lab, membership2
    #else:
        #return lab


#def _calc_TargetEffLaplacian_interchr_20160802(pars):
    #"""
    #Get effective Laplacian between targets: inter-chromosomal case.
    #Requires pars: chrfullname1/2, ntarget1/2, beta, norm.
    #Formula for a in c1, b in c2:
        #l_{a,b} = sum_{i, j in c1 union c2} {q_a(i) pi_i L_{i,j} q_b(j)}
    #"""
    #tpars = copy.deepcopy(pars)
    #chrfullname1 = pars['chrfullname1']
    #chrfullname2 = pars['chrfullname2']
    #ntarget1 = pars['ntarget1']
    #ntarget2 = pars['ntarget2']
    #norm = pars['norm']
    ## Get tsets, committors
    #tpars['chrfullname'] = chrfullname1
    #mappedtset1 = _get_tsetReference_20180802(tpars, ntarget1)[0]
    #tpars['chrref'] = chrfullname1
    #mapping, _ = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    #tset1 = np.array([list(mapping).index(t) for t in mappedtset1])
    #qAc1 = _get_TargetCommittor(_get_runbinarydir(tpars), tset1,
                    #norm=norm)
    #tpars['chrfullname'] = chrfullname2
    #mappedtset2 = _get_tsetReference_20180802(tpars, ntarget2)[0]
    #tpars['chrref'] = chrfullname2
    #mapping, _ = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    #tset2 = np.array([list(mapping).index(t) for t in mappedtset2])
    #qBc2 = _get_TargetCommittor(_get_runbinarydir(tpars), tset2,
                    #norm=norm)
    ## Get trans-fij, targetmembership
    #tpars['chrref'] = chrfullname1
    #fmat1, _, _, md1 = _get_arrays(_get_runbinarydir(tpars), norm=norm)
    #tpars['chrref'] = chrfullname2
    #fmat2, _, _, md2 = _get_arrays(_get_runbinarydir(tpars), norm=norm)
    #pars['chrref'] = '_'.join([chrfullname1, chrfullname2])
    #fmat12, mdp1, mdp2 = _get_fmatmap_inter(pars)
    #targetmembership = _targetmembership_softpartition_2chr(
                    #qAc1, qBc2, md1, md2)
    ## Compute lac1bc2
    ### Trans-lij
    #nbins1, nbins2 = md1[1], md2[1]
    #fillmat1 = plu._build_fullarray_inter(fmat1, md1, md1, np.nan)
    #fillmat12 = plu._build_fullarray_inter(fmat12, mdp1, mdp2, np.nan)
    #fillmat2 = plu._build_fullarray_inter(fmat2, md2, md2, np.nan)
    #fmatcombined = np.array(np.bmat([[fillmat1, fillmat12],
                 #[fillmat12.T, fillmat2]]))
    #del fillmat1, fillmat12, fillmat2
    #fmatcombined2 = fmatcombined.copy()
    #fmatcombined2[np.isnan(fmatcombined)] = 0.0
    #del fmatcombined
    #pivec = np.sum(fmatcombined2, axis=1)
    #pmatcombined = np.array([v / pivec[i] if pivec[i] > 0.0 else 0.0 * v
                    #for i, v in enumerate(fmatcombined2)])
    #pivec /= np.sum(pivec)
    #lmatcombined = pmatcombined - np.diag(np.ones_like(pivec))
    #del fmatcombined2, pmatcombined
    ### lac1bc2
    #lac1bc2 = np.zeros((ntarget1, ntarget2))
    #for a1 in range(ntarget1):
        #for b2 in range(ntarget2):
            #v1 = targetmembership[a1] * pivec
            #v2 = targetmembership[b2 + ntarget1]
            #filtermat = np.outer(v1, v2)
            #lac1bc2[a1, b2] = np.sum(filtermat * lmatcombined)
    #return lac1bc2


#def _calc_TargetEffLaplacian_interchr_20160829(pars):
    #"""
    #Get effective Laplacian between targets: inter-chromosomal case.
    #Requires pars: chrfullname1/2, ntarget1/2, beta, norm.
    #Note: Uses fmat from beta = 1.0
    #Formula for a in c1, b in c2:
        #l_{a,b} = sum_{i, j in c1 union c2} {q_a(i) pi_i L_{i,j} q_b(j)}
    #"""
    #tpars = copy.deepcopy(pars)
    #chrfullname1 = pars['chrfullname1']
    #chrfullname2 = pars['chrfullname2']
    #ntarget1 = pars['ntarget1']
    #ntarget2 = pars['ntarget2']
    #norm = pars['norm']
    #beta = pars['beta']
    ## Get tsets, committors
    #tpars['chrfullname'] = chrfullname1
    #mappedtset1 = _get_tsetReference_20180802(tpars, ntarget1)[0]
    #tpars['chrref'] = chrfullname1
    #mp1 = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    #tset1 = np.array([list(mp1[0]).index(t) for t in mappedtset1])
    #qAc1 = _get_TargetCommittor(_get_runbinarydir(tpars), tset1,
                    #norm=norm)
    #tpars['chrfullname'] = chrfullname2
    #mappedtset2 = _get_tsetReference_20180802(tpars, ntarget2)[0]
    #tpars['chrref'] = chrfullname2
    #mp2 = _get_mappingdata(_get_runbinarydir(tpars), norm=norm)
    #tset2 = np.array([list(mp2[0]).index(t) for t in mappedtset2])
    #qBc2 = _get_TargetCommittor(_get_runbinarydir(tpars), tset2,
                    #norm=norm)
    ## Get trans-fij, targetmembership
    #tpars['beta'] = 1.0
    #tpars['chrref'] = chrfullname1
    #fmat1, _, _, md1 = _get_arrays(_get_runbinarydir(tpars), norm=norm)
    #tpars['chrref'] = chrfullname2
    #fmat2, _, _, md2 = _get_arrays(_get_runbinarydir(tpars), norm=norm)
    #pars['chrref'] = '_'.join([chrfullname1, chrfullname2])
    #fmat12, mdp1, mdp2 = _get_fmatmap_inter(pars)
    #targetmembership = _targetmembership_softpartition_2chr(
                    #qAc1, qBc2, mp1, mp2)
    ## Compute lac1bc2
    ### Trans-lij
    #nbins1, nbins2 = md1[1], md2[1]
    #fillmat1 = plu._build_fullarray_inter(fmat1, md1, md1, np.nan)
    #fillmat12 = plu._build_fullarray_inter(fmat12, mdp1, mdp2, np.nan)
    #fillmat2 = plu._build_fullarray_inter(fmat2, md2, md2, np.nan)
    #fmatcombined = np.array(np.bmat([[fillmat1, fillmat12],
                 #[fillmat12.T, fillmat2]]))
    #del fillmat1, fillmat12, fillmat2
    #fmatcombined2 = fmatcombined.copy()
    #fmatcombined2[np.isnan(fmatcombined)] = 0.0
    #del fmatcombined
    #pivec = np.sum(fmatcombined2, axis=1)
    #pmatcombined = np.array([v / pivec[i] if pivec[i] > 0.0 else 0.0 * v
                    #for i, v in enumerate(fmatcombined2)])
    #pivec /= np.sum(pivec)
    #lmatcombined = pmatcombined - np.diag(np.ones_like(pivec))
    #del fmatcombined2, pmatcombined
    ### lac1bc2
    #lac1bc2 = np.zeros((ntarget1, ntarget2))
    #for a1 in range(ntarget1):
        #for b2 in range(ntarget2):
            #v1 = targetmembership[a1] * pivec
            #v2 = targetmembership[b2 + ntarget1]
            #filtermat = np.outer(v1, v2)
            #lac1bc2[a1, b2] = np.sum(filtermat * lmatcombined)
    #return lac1bc2


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


#def _targetmembership_splitcenterpartition(pars, tset, mappingdata):
    #"""
    #Compute target membership function.
    #Split-center partitioning: assign each locus to the nearest target
                               #along genome.
    #"""
    ## Force sorting
    #tset2 = np.sort(tset)
    #ntarget = len(tset)
    #mapping, nbins = mappingdata
    #targetmembership = np.zeros((ntarget, nbins))
    #for i in range(nbins):
        #if i not in mapping:
            #continue
        #ind = np.where(mapping == i)
        #distances = np.abs(ind - tset2)
        #targetmembership[np.argmin(distances), i] = 1.0
    #return targetmembership


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

