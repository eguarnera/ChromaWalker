#!/usr/bin/env python
"""
Collection of plotting functions and utilities for Hi-C analysis

Part of ChromaWalker package
"""


import numpy as np
import os
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
import hicutils as hcu
import networkx as nx
import numpy.ma as ma
import matplotlib.patches as patches

#####################################################
# Utilities
## Padding to full arrays (including omitted data rows)


def _build_fullvector(data, mappingdata, fillval):
    """
    Build full array from mapped subvector,
    filling empty elements with fillval.
    """
    mapping, nbins = mappingdata
    datapad = np.zeros(nbins)
    datapad.fill(fillval)
    datapad[mapping] = data
    return datapad


def _build_fullarray(data, mappingdata, fillval):
    """
    Build full array from mapped subarray,
    filling empty rows and columns with fillval.
    """
    mapping, nbins = mappingdata
    datapad = np.zeros((nbins, nbins))
    datapad.fill(fillval)
    for i, vec in enumerate(data):
        datapad[mapping[i], mapping] = vec
    return datapad


def _build_fullarray_inter(data, mappingdata1, mappingdata2, fillval):
    """
    Build full array from mapped subarray,
    filling empty rows and columns with fillval.
    Inter-chromosomal mode: x- and y-axes have different mappingdata.
    """
    mapping1, nbins1 = mappingdata1
    mapping2, nbins2 = mappingdata2
    datapad = np.zeros((nbins1, nbins2))
    datapad.fill(fillval)
    for i, vec in enumerate(data):
        datapad[mapping1[i], mapping2] = vec
    return datapad


def _extendarray(datapack, membership, membermode='hard'):
    """
    Form an extended-array view of a reduced data matrix.
    Input:
        datapack: (M, M) reduced array of data values
        membership: (N,) or (M, N) array of membership values
                    (N,) for 'hard' membership,
                    (M, N) for 'soft' membership
        membermode: 'hard' for assigning each extended pixel to one
                    reduced-data pixel
                    Note: For extended pixels which should have no data,
                          set membership value to negative number.
                    'soft' to assign weights for each reduced-data pixel,
                    for each extended pixel
    Returns:
        extarray: Extended array
    """
    # Create soft-type membership function
    npack = len(datapack)
    if membermode == 'hard':
        membfunc = np.array([membership == i for i in range(npack)])
    elif membermode == 'soft':
        membfunc = membership.copy()
    else:
        print 'Invalid membership mode %s!' % membership
        sys.exit(1)
    nbins, _ = membfunc.shape
    # Create extended array
    extarray = np.dot(np.dot(membfunc.T, datapack), membfunc)
    return extarray


#####################################################
# Plotting functions


def _plot_cytobands(cytobanddata, resolution, x, plotbname=False):
    cytoband, blimlist, bnamelist, _ = cytobanddata
    if plotbname:
        x2 = x.twiny()
    chrlen = len(cytoband) * resolution / 1e6
    bands = np.array([cytoband] * 10) * 1.0
    cmap = plt.get_cmap('binary')
    cmap.set_under((1.0, 0.0, 0.0))
    x.imshow(bands, interpolation='nearest', cmap=cmap,
                    vmin=0.0, vmax=4.0, aspect='auto', extent=[0, chrlen, 0, 1])
    x.set_xlim(0, chrlen)
    x.set_ylim(0, 1)
    x.set_yticks(())
    if plotbname:
        x2.set_xlim(0, len(cytoband))
        x2.set_ylim(0, 1)
        bandctrs = [(i + j) / (2.0) for i, j in blimlist]
        x2.set_xticks(bandctrs)
        x2.set_xticklabels(bnamelist, rotation=45)
    return


def _plot_cytobands_nchr(cytobanddatas, resolution, x, plotbname=False):
    #nchrs = len(cytobanddatas)
    cytoband = []
    blimlist = []
    bnamelist = []
    shifts = []
    for ct, bl, bn, _ in cytobanddatas:
        shift = len(cytoband) * resolution / 1.0e6
        shifts.append(shift)
        cytoband.extend(list(ct))
        blimlist.extend(list(np.array(bl) + shift))
        bnamelist.extend(list(bn))
    #cytoband, blimlist, bnamelist, _ = cytobanddata
    if plotbname:
        x2 = x.twiny()
    chrlen = len(cytoband) * resolution / 1e6
    bands = np.array([cytoband] * 10) * 1.0
    cmap = plt.get_cmap('binary')
    cmap.set_under((1.0, 0.0, 0.0))
    print bands.shape
    x.imshow(bands, interpolation='nearest', cmap=cmap,
                    vmin=0.0, vmax=4.0, aspect='auto', extent=[0, chrlen, 0, 1])
    x.set_xlim(0, chrlen)
    x.set_ylim(0, 1)
    x.set_yticks(())
    for shift in shifts:
        x.axvline(x=shift, c='b')
    if plotbname:
        x2.set_xlim(0, len(cytoband))
        x2.set_ylim(0, 1)
        bandctrs = [(i + j) / (2.0) for i, j in blimlist]
        x2.set_xticks(bandctrs)
        x2.set_xticklabels(bnamelist, rotation=45)
    return


def _plot_cytobands_vert(cytobanddata, resolution, x, plotbname=False):
    cytoband, blimlist, bnamelist, _ = cytobanddata
    if plotbname:
        x2 = x.twiny()
    chrlen = len(cytoband) * resolution / 1e6
    bands = np.array([cytoband] * 10) * 1.0
    cmap = plt.get_cmap('binary')
    cmap.set_under((1.0, 0.0, 0.0))
    x.imshow(bands.T, interpolation='nearest', cmap=cmap,
                    vmin=0.0, vmax=4.0, aspect='auto', extent=[0, 1, chrlen, 0])
    x.set_ylim(chrlen, 0)
    x.set_xlim(0, 1)
    x.set_xticks(())
    if plotbname:
        x2.set_xlim(0, len(cytoband))
        x2.set_ylim(0, 1)
        bandctrs = [(i + j) / (2.0) for i, j in blimlist]
        x2.set_yticks(bandctrs)
        x2.set_yticklabels(bnamelist, rotation=45)
    return


def _plot_cytobands_nchrvert(cytobanddatas, resolution, x, plotbname=False):
    #nchrs = len(cytobanddatas)
    cytoband = []
    blimlist = []
    bnamelist = []
    shifts = []
    for ct, bl, bn, _ in cytobanddatas:
        shift = len(cytoband) * resolution / 1.0e6
        shifts.append(shift)
        cytoband.extend(list(ct))
        blimlist.extend(list(np.array(bl) + shift))
        bnamelist.extend(list(bn))
    #cytoband, blimlist, bnamelist, _ = cytobanddata
    if plotbname:
        x2 = x.twiny()
    chrlen = len(cytoband) * resolution / 1e6
    bands = np.array([cytoband] * 10) * 1.0
    cmap = plt.get_cmap('binary')
    cmap.set_under((1.0, 0.0, 0.0))
    print bands.shape
    x.imshow(bands.T, interpolation='nearest', cmap=cmap,
                    vmin=0.0, vmax=4.0, aspect='auto', extent=[0, 1, chrlen, 0])
    x.set_ylim(chrlen, 0)
    x.set_xlim(0, 1)
    x.set_xticks(())
    for shift in shifts:
        x.axhline(y=shift, c='b')
    if plotbname:
        x2.set_xlim(0, len(cytoband))
        x2.set_ylim(0, 1)
        bandctrs = [(i + j) / (2.0) for i, j in blimlist]
        x2.set_yticks(bandctrs)
        x2.set_yticklabels(bnamelist, rotation=45)
    return


def _plot_heatmap(data, x, res, cmap=None,
                n=1.0, scale='pow', extent=None, vlims=None,
                badcolor=(1.0, 1.0, 1.0, 0.0), returnimg=False):
    """
    Pretty plotting of interaction heatmaps.
    """
    if extent is None:
        extent = [0, len(data) * res / 1e6, len(data) * res / 1e6, 0]
    if cmap is None:
        cmap = plt.get_cmap('jet')
    if vlims is None:
        vlims = [None, None]
        cmap.set_bad(badcolor)
    if scale == 'pow':
        d = np.abs(data) ** n * np.sign(data)
        img = x.imshow(d, interpolation='nearest', cmap=cmap,
                        extent=extent, vmin=vlims[0], vmax=vlims[1], alpha=1.0)
    elif scale == 'log':
        img = x.imshow(np.log(data), interpolation='nearest',
                        cmap=cmap, extent=extent, vmin=vlims[0], vmax=vlims[1],
                        alpha=1.0)
    else:
        print 'Invalid 2D plotting scale!'
    return img if returnimg else None


def _plot_heatmap_uneven(data, edges, x, res, cmap=None,
                n=1.0, scale='pow', extent=None, vlims=None,
                badcolor=(1.0, 1.0, 1.0, 0.0), returnimg=False):
    """
    Pretty plotting of interaction heatmaps.
    """
    if extent is None:
        extent = [0, len(data) * res / 1e6, len(data) * res / 1e6, 0]
    if cmap is None:
        cmap = plt.get_cmap('jet')
    if vlims is None:
        vals = data.flatten()
        vlims = [np.min(vals[np.isfinite(vals)]),
                 np.max(vals[np.isfinite(vals)])]
        cmap.set_bad(badcolor)
    if scale == 'pow':
        d = np.abs(data) ** n * np.sign(data)
        dm = ma.masked_where(np.isnan(d), d)
        img = x.pcolormesh(edges, edges, dm, cmap=cmap,
                        vmin=vlims[0] ** n, vmax=vlims[1] ** n, alpha=1.0,
                        edgecolors='face')
    elif scale == 'log':
        d = np.log(data)
        dm = ma.masked_where(np.isnan(d), d)
        img = x.pcolormesh(edges, edges, dm,
                        cmap=cmap, vmin=np.log(vlims[0]), vmax=np.log(vlims[1]),
                        alpha=1.0, edgecolors='face')
    else:
        print 'Invalid 2D plotting scale!'
    return img if returnimg else None


def _fillplot_heatmap_2bands(x, fmat, mappingdata, blimlist, res, cmap=None,
                n=1.0, scale='pow', extent=None, vlims=None,
                badcolor=(1.0, 1.0, 1.0, 0.0)):
    """
    Plot two non-adjacent bands.
    Input:
        fmat: interaction matrix, with empty rows removed
        mappingdata: Maps fmat pixels to full matrix pixel positions
        blimlist: Band boundaries in bp
        res: Resolution of full matrix
    """
    # Pad array to fill band ranges
    ## Allocate padded array
    (stpos1, enpos1), (stpos2, enpos2) = blimlist[:2]
    nbins1 = (enpos1 - stpos1) / res
    nbins2 = (enpos2 - stpos2) / res
    nbins12 = nbins1 + nbins2
    fmatpad = np.zeros((nbins12, nbins12))
    ## Map fmat indices to fmatpad indices
    padpos = list(np.arange(nbins1) * res + stpos1) + \
                    list(np.arange(nbins2) * res + stpos2)
    maptopad = []
    mapmask = []
    for mapval in mappingdata[0]:
        thispos = mapval * res
        if thispos in padpos:
            maptopad.append(padpos.index(thispos))
            mapmask.append(True)
        else:
            mapmask.append(False)
    mapmask = np.array(mapmask)
    maptopad = np.array(maptopad)
    ## Fill
    for i, m in enumerate(maptopad):
        fmatpad[m, maptopad] = fmat[i, mapmask]
    print np.max(fmatpad), np.min(fmatpad[fmatpad > 0.0])
    print np.product(fmatpad.shape), np.sum(np.nonzero(fmatpad))
    # Plot padded array
    if extent is None:
        extent = [0, len(fmatpad) * res / 1e6, len(fmatpad) * res / 1e6, 0]
    if cmap is None:
        cmap = plt.get_cmap('jet')
    if vlims is None:
        vlims = [None, None]
        cmap.set_bad(badcolor)
    if scale == 'pow':
        d = np.abs(fmatpad) ** n * np.sign(fmatpad)
        x.imshow(d, interpolation='nearest', cmap=cmap,
                        extent=extent, vmin=vlims[0], vmax=vlims[1], alpha=1.0)
    elif scale == 'log':
        x.imshow(np.log(fmatpad), interpolation='nearest',
                        cmap=cmap, extent=extent, vmin=vlims[0], vmax=vlims[1],
                        alpha=1.0)
    # Draw boundary line
    x.axvline(x=nbins1 * res / 1.0e6, c='k')
    x.axhline(y=nbins1 * res / 1.0e6, c='k')
    # Set ticks at 1Mbp interval
    maxticktrial = int(np.max(padpos) / 1.0e6)
    tickpos = []
    ticklabel = []
    for i in range(maxticktrial + 1):
        trialpos = int(i * 1.0e6)
        if trialpos in padpos:
            tickpos.append(padpos.index(trialpos) * res / 1e6)
            ticklabel.append(i)
    x.set_xticks(tickpos)
    x.set_xticklabels(ticklabel)
    x.set_yticks(tickpos)
    x.set_yticklabels(ticklabel)
    return


def _plot_tsetlab(x, tset, lab, mappingdata, res, scale='pow',
                n=1.0, cmap=None, s=None):
    """
    2D scatter plot of inter-target interactions.
    """
    # Unpack data
    mapping, nbins = mappingdata
    if s is None:
        s = 12
    # Rescale values
    lab2 = lab.copy()
    lab2[lab == 0.0] = np.nan
    labrescale = (lab2 - np.min(lab2[lab2 > 0.0])) / (np.max(lab2[lab2 > 0.0]) -
                    np.min(lab2[lab2 > 0.0]))
    if scale == 'log':
        labrescale = np.log(labrescale)
    else:
        labrescale = labrescale ** n
    if cmap is None:
        cmap = plt.get_cmap('jet')
    clrarray = cmap(labrescale)
    # Plot
    tsetsort = np.sort(tset)
    nt = len(lab)
    for i in range(nt):
        xval = mapping[tsetsort[i]] * res / 1.0e6
        for j in range(nt):
            yval = mapping[tsetsort[j]] * res / 1.0e6
            if i == j:
                x.scatter(xval, yval, marker='+', c='k', s=20)
            else:
                x.scatter(xval, yval,
                                marker='o', c=clrarray[i, j], s=s, lw=0,
                                                alpha=1.0)
    x.set_axis_bgcolor((0.8, 0.8, 0.8))
    x.set_xlim(0, nbins * res / 1.0e6)
    x.set_ylim(nbins * res / 1.0e6, 0)


def _scatterplot_RefLab_2bands(x, pars, blimlist, resvals=(100000, 200000)):
    """
    Plot scatter plot of inter-target effective interactions.
    Zooms into region between two bands.
    Input:
        resvals: (res_cytoband, res_tset)
                 Default value: 100kb, 200kb
    """
    res_cytoband, res_tset = resvals
    (stpos1, enpos1), (stpos2, enpos2) = blimlist[:2]
    # Get reference tset
    pars_t = copy.deepcopy(pars)
    pars_t['res'] = res_tset
    tset = hcu._get_tsetReference(pars_t, refdate='20160610')
    tsetsort = np.sort(tset)
    # Get targets in chosen region(s)
    (stpos1, enpos1), (stpos2m, enpos2) = blimlist[:2]
    ## Get mappingdata
    mappingdata = hcu._get_mappingdata(hcu._get_runbinarydir(pars_t))
    mapping, nbins = mappingdata
    tgtsel = []
    tgtpos = []
    for i, t in enumerate(tsetsort):
        tpos = mapping[t] * res_tset
        if (tpos >= stpos1 and tpos < enpos1) or \
                        (tpos >= stpos2 and tpos < enpos2):
            tgtsel.append(i)
            tgtpos.append(tpos / 1.0e6)
    # Get lab
    lab = hcu._get_TargetEffLaplacian(hcu._get_runbinarydir(pars_t), tset)
    cmap = plt.get_cmap('jet')
    labrescale = (lab - np.min(lab[lab > 0.0])) / (np.max(lab) -
                np.min(lab[lab > 0.0]))
    clrarray = cmap(labrescale)
    x1 = []
    x2 = []
    p1 = []
    p2 = []
    print tgtsel
    print tgtpos
    for i1, t1 in enumerate(tgtsel):
        for i2, t2 in enumerate(tgtsel):
            if i2 == i1:
                continue
            x1.append(t1)
            x2.append(t2)
            if tgtpos[i1] > enpos1 / 1.0e6:
                p1.append(tgtpos[i1] + (enpos1 - stpos1 - stpos2 +
                                res_tset / 2) / 1.0e6)
            else:
                p1.append(tgtpos[i1] - (stpos1 - res_tset / 2) / 1.0e6)
            if tgtpos[i2] > enpos1 / 1.0e6:
                p2.append(tgtpos[i2] + (enpos1 - stpos1 - stpos2 +
                                res_tset / 2) / 1.0e6)
            else:
                p2.append(tgtpos[i2] - (stpos1 - res_tset / 2) / 1.0e6)
    # Scatter plot, broken axes
    x.scatter(p1, p2, marker='o', c=clrarray[x1, x2], lw=0.5)
    x.set_axis_bgcolor((0.8, 0.8, 0.8))
    x.axvline(x=(enpos1 - stpos1) / 1.0e6, c='k')
    x.axhline(y=(enpos1 - stpos1) / 1.0e6, c='k')
    # Set ticks at 1Mbp interval
    maxticktrial = int(np.max(blimlist) / 1.0e6)
    tickpos = []
    ticklabel = []
    for i in range(maxticktrial + 1):
        trialpos = int(i * 1.0e6)
        if (trialpos >= stpos1 and trialpos < enpos1):
            tickpos.append((trialpos - stpos1) / 1.0e6)
            ticklabel.append(i)
        elif (trialpos >= stpos2 and trialpos < enpos2):
            tickpos.append((trialpos - stpos2 + enpos1 - stpos1) / 1.0e6)
            ticklabel.append(i)
    x.set_xticks(tickpos)
    x.set_xticklabels(ticklabel)
    x.set_yticks(tickpos)
    x.set_yticklabels(ticklabel)
    # Draw boundary line
    lims = 0, (enpos2 - stpos2 +
                    enpos1) / 1.0e6 - ticklabel[0] - stpos1 / 1.0e6 \
                    + ticklabel[0]
    print lims
    x.set_xlim(lims)
    x.set_ylim(lims[::-1])
    x.set_aspect(1)


def _plot_qAi(mat, tset, ax, mapping, res, shift, cmap=cm.Dark2):
    tsettemp = np.sort(tset)
    colors = map(cmap, mapping[np.array(tsettemp)] / np.max(mapping * 1.0))
    for i, tgt in enumerate(tsettemp):
        clr = colors[i]
        xvals = np.arange(np.max(mapping) + 1) * res / 1.0e6
        yvals = np.ones_like(xvals, dtype='float64') * np.nan
        yvals[mapping] = mat[i] + shift
        _ = ax.plot(xvals, yvals, c=clr)
        _ = ax.set_xlim(0, np.max(xvals))


def _plot_qAi_max(mat, tset, ax, mapping, res, shift, cmap=cm.Dark2):
    tsettemp = np.sort(tset)
    colors = map(cmap, mapping[np.array(tsettemp)] / np.max(mapping * 1.0))
    for i, tgt in enumerate(tsettemp):
        clr = colors[i]
        xvals = np.arange(np.max(mapping) + 1) * res / 1.0e6
        yvals = np.ones_like(xvals, dtype='float64') * np.nan
        yvals[mapping] = mat[i] + shift
        mvals = np.ones_like(xvals, dtype='float64') * np.nan
        mvals[mapping] = np.max(mat, axis=0) + shift
        maxmask = (yvals == mvals)
        yvals[np.array(1 - maxmask, dtype='bool')] = np.nan
        #_ = ax.plot(xvals[maxmask], yvals[maxmask], c=clr)
        _ = ax.plot(xvals, yvals, c=clr)
        _ = ax.set_xlim(0, np.max(xvals))
    xvals = np.arange(np.max(mapping) + 1) * res / 1.0e6
    yvals = np.ones_like(xvals, dtype='float64') * np.nan
    yvals[mapping] = np.max(mat, axis=0) + shift
    #_ = ax.plot(xvals, yvals, 'b')
    _ = ax.axhline(y=shift, color='k')
    _ = ax.set_xlim(0, np.max(xvals))


def _plot_qAi_tgtstrip(tset, ax, mapping, res, shift, cmap=cm.Dark2,
                lw=1, ls='-', c=None):
    targets = []
    for i in tset:
        targets.append(i)
    targets.sort()
    if c is None:
        colors = map(cmap, mapping[np.array(targets)] / np.max(mapping * 1.0))
    else:
        colors = ['k'] * len(targets)
    for i, tgt in enumerate(targets):
        clr = colors[i]
        xvals = [mapping[tgt] * res / 1.0e6] * 2
        yvals = shift, shift + 1.0
        _ = ax.plot(xvals, yvals, c=clr, lw=lw, ls=ls)
    xvals = np.arange(np.max(mapping) + 1) * res / 1.0e6
    _ = ax.axhline(y=shift, color='k')
    _ = ax.set_xlim(0, np.max(xvals))


def _plot_qAi_fill(qAi, tset, ax, mapping, res, shift, cmap=cm.Dark2):
    targets = [v for v in qAi]
    targets.sort()
    ntarget = len(targets)
    mat = np.array([qAi[k] for k in targets])
    ntarget = len(tset)
    # Cumulative qAi
    qp = [np.sum(mat[:i + 1], axis=0) for i in range(ntarget)]
    qp = [np.zeros_like(qp[0])] + qp
    qp = np.array(qp)
    centroids = [(qAi[k] * mapping) / np.sum(qAi[k]) for t in targets]
    #colors = map(cmap, mapping[np.array(targets)] / np.max(mapping * 1.0))
    colors = map(cmap, mapping[np.array(centroids)] / np.max(mapping * 1.0))
    xvals = np.arange(np.max(mapping) + 1) * res / 1.0e6
    yvals1 = np.ones_like(xvals, dtype='float64') * np.nan
    yvals1[mapping] = qp[0] + shift
    for i in range(ntarget):
        yvals1 = np.ones_like(xvals, dtype='float64') * np.nan
        yvals2 = np.ones_like(xvals, dtype='float64') * np.nan
        yvals1[mapping] = qp[i] + shift
        yvals2[mapping] = qp[i + 1] + shift
        ax.fill_between(xvals, yvals1, yvals2, facecolor=colors[i],
                        where=(yvals2 > yvals1), lw=0)
    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(xvals), np.max(xvals))


def _plot_qAi_fill_hardp(qAi, tset, ax, mapping, res, shift, cmap=cm.Dark2):
    targets = [v for v in qAi]
    targets.sort()
    ntarget = len(targets)
    mat = np.array([qAi[k] for k in targets])
    mat2 = mat.copy()
    for i in range(len(mat2.T)):
        mat2[:, i] = 0.0
        mat2[np.argmax(mat[:, i]), i] = 1.0
    ntarget = len(tset)
    # Cumulative qAi
    qp = [np.sum(mat2[:i + 1], axis=0) for i in range(ntarget)]
    qp = [np.zeros_like(qp[0])] + qp
    qp = np.array(qp)
    centroids = [np.sum(qAi[t] * mapping) / np.sum(qAi[t]) for t in targets]
    #print np.min(np.interp(centroids, np.arange(len(mapping)), mapping,
                    #left=mapping[0], right=mapping[-1]) / np.max(mapping * 1.0)), \
            #np.max(np.interp(centroids, np.arange(len(mapping)), mapping,
                    #left=mapping[0], right=mapping[-1]) / np.max(mapping * 1.0)), \
                    #centroids
    #_ = raw_input('...:')
    #colors = map(cmap, mapping[np.array(targets)] / np.max(mapping * 1.0))
    #colors = map(cmap, np.interp(centroids, np.arange(len(mapping)), mapping,
                    #left=mapping[0], right=mapping[-1]) / np.max(mapping * 1.0))
    if cmap is None:
        colors = [(0.5, 0.5, 0.5) for c in centroids]
    else:
        colors = map(cmap, centroids / np.max(mapping * 1.0))
    xvals = np.arange(np.max(mapping) + 1) * res / 1.0e6
    yvals1 = np.ones_like(xvals, dtype='float64') * np.nan
    yvals1[mapping] = qp[0] + shift
    for i in range(ntarget):
        yvals1 = np.ones_like(xvals, dtype='float64') * np.nan
        yvals2 = np.ones_like(xvals, dtype='float64') * np.nan
        yvals1[mapping] = qp[i] + shift
        yvals2[mapping] = qp[i + 1] + shift
        ax.fill_between(xvals, yvals1, yvals2, facecolor=colors[i],
                        where=(yvals2 > yvals1), lw=0.0)
    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(xvals), np.max(xvals))


def _plot_qAi_fill_hardp2(qAi, tset, ax, mapping, res, shift, cmap=cm.Dark2):
    targets = [v for v in qAi]
    targets.sort()
    ntarget = len(targets)
    mat = np.array([qAi[k] for k in targets])
    mat2 = mat.copy()
    mat3 = np.zeros(len(mat[0]), dtype=int) - 1
    for i in range(len(mat2.T)):
        mat2[:, i] = 0.0
        mat2[np.argmax(mat[:, i]), i] = 1.0
        mat3[i] = np.argmax(mat[:, i])
    ntarget = len(tset)
    # Cumulative qAi
    qp = [np.sum(mat2[:i + 1], axis=0) for i in range(ntarget)]
    qp = [np.zeros_like(qp[0])] + qp
    qp = np.array(qp)
    centroids = [np.sum(qAi[t] * mapping) / np.sum(qAi[t]) for t in targets]
    if cmap is None:
        colors = [(0.5, 0.5, 0.5) for c in centroids]
    else:
        colors = map(cmap, centroids / np.max(mapping * 1.0))
    xvals = np.arange(np.max(mapping) + 1) * res / 1.0e6
    yvals1 = np.ones_like(xvals, dtype='float64') * np.nan
    yvals1[mapping] = qp[0] + shift
    blockedges = []
    for i in range(ntarget):
        yvals1 = np.ones_like(xvals, dtype='float64') * np.nan
        yvals2 = np.ones_like(xvals, dtype='float64') * np.nan
        yvals1[mapping] = qp[i] + shift
        yvals2[mapping] = qp[i + 1] + shift
        ax.fill_between(xvals, yvals1, yvals2, facecolor=colors[i],
                        where=(yvals2 > yvals1), lw=0)
        thisblock = mat2[i]
        thisblockedgeL = np.nonzero((thisblock[:-1] - thisblock[1:]) < 0)[0]
        for edgeL in thisblockedgeL:
            if mat3[edgeL] >= 0 and mat3[edgeL + 1] >= 0:
                xvs = [mapping[edgeL + 1] * res / 1.0e6] * 2
                yvs = [shift, shift + 1]
                ax.plot(xvs, yvs, 'k')
        blockedges.append(list(thisblockedgeL))
    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(xvals), np.max(xvals))
    print "block edges:"
    print blockedges


def _plot_qAi_fill_hardp2_ntgrad(qAi, tset, ax, mapping, res, shift,
                cmap=cm.Blues, ntmax=50):
    targets = [v for v in qAi]
    targets.sort()
    ntarget = len(targets)
    mat = np.array([qAi[k] for k in targets])
    mat2 = mat.copy()
    mat3 = np.zeros(len(mat[0]), dtype=int) - 1
    for i in range(len(mat2.T)):
        mat2[:, i] = 0.0
        mat2[np.argmax(mat[:, i]), i] = 1.0
        mat3[i] = np.argmax(mat[:, i])
    ntarget = len(tset)
    # Cumulative qAi
    qp = [np.sum(mat2[:i + 1], axis=0) for i in range(ntarget)]
    qp = [np.zeros_like(qp[0])] + qp
    qp = np.array(qp)
    centroids = [np.sum(qAi[t] * mapping) / np.sum(qAi[t]) for t in targets]
    if cmap is None:
        colors = [(0.5, 0.5, 0.5) for c in centroids]
    else:
        p = 0.3
        clr = (ntarget ** p - ntmax ** p) / (2.0 ** p - ntmax ** p)
        colors = [cmap(clr)] * len(centroids)
    xvals = np.arange(np.max(mapping) + 1) * res / 1.0e6
    yvals1 = np.ones_like(xvals, dtype='float64') * np.nan
    yvals1[mapping] = qp[0] + shift
    blockedges = []
    for i in range(ntarget):
        yvals1 = np.ones_like(xvals, dtype='float64') * np.nan
        yvals2 = np.ones_like(xvals, dtype='float64') * np.nan
        yvals1[mapping] = qp[i] + shift
        yvals2[mapping] = qp[i + 1] + shift
        ax.fill_between(xvals, yvals1, yvals2, facecolor=colors[i],
                        where=(yvals2 > yvals1), lw=0)
        thisblock = mat2[i]
        thisblockedgeL = np.nonzero((thisblock[:-1] - thisblock[1:]) < 0)[0]
        for edgeL in thisblockedgeL:
            if mat3[edgeL] >= 0 and mat3[edgeL + 1] >= 0:
                xvs = [mapping[edgeL + 1] * res / 1.0e6] * 2
                yvs = [shift, shift + 1]
                ax.plot(xvs, yvs, 'k')
        blockedges.append(list(mapping[thisblockedgeL]))
    print blockedges
    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(xvals), np.max(xvals))


def _plot_PartitionsHierarchy(ax, limslist, pieceidlist, tclist, ntlist,
                levels, res):
    """
    Plot hierarchy of partitions in rectangular box format.
    Selected ntargets in ntlist
    For ntarget in levels, bars are twice as big.
    """
    ytickpos = []
    for i, (lims2, pieceids2, tc, ntarget) in \
                    enumerate(zip(limslist, pieceidlist, tclist, ntlist)):
        shift = len(ntlist) - i - 1 + np.sum(ntarget < np.array(levels))
        height = 2.0 if ntarget in levels else 1.0
        ytickpos.append(shift + 0.5 * height)
        mypatches = []
        for (st, en), pieceid in zip(lims2, pieceids2):
            if pieceid < 0:
                mypatches.append(patches.Rectangle(
                                 (st * res / 1.0e6, shift),
                                 (en - st) * res / 1.0e6, height, fill=False,
                                 linewidth=0.5))
            else:
                mypatches.append(patches.Rectangle(
                                 (st * res / 1.0e6, shift),
                                 (en - st) * res / 1.0e6, height,
                                 facecolor=tc, linewidth=0.2))
        for p in mypatches:
            ax.add_patch(p)
        yp = [shift, shift + height]
        for (st, en) in lims2:
            xp = [st * res / 1.0e6, st * res / 1.0e6]
            ax.plot(xp, yp, 'k', lw=1)
            xp = [en * res / 1.0e6, en * res / 1.0e6]
            ax.plot(xp, yp, 'k', lw=1)
    ax.set_yticks(ytickpos)
    ax.set_yticklabels(ntlist)
    ax.set_xlim(0, np.max(map(max, lims2)) * res / 1.0e6)
    ax.set_ylim(0, len(ntlist) + np.sum(ntarget < np.array(levels)))


def _plot_hcluster_fill_hardp(hc, ax, mapping, res, shift, cmap=cm.Dark2):
    mat3 = hc.copy()
    ntarget = np.max(mat3) + 1
    mat2 = np.array([map(int, mat3 == i) for i in range(ntarget)])
    # Cumulative qAi
    qp = [np.sum(mat2[:i + 1], axis=0) for i in range(ntarget)]
    qp = [np.zeros_like(qp[0])] + qp
    qp = np.array(qp)
    centroids = [np.sum((mat3 == i) * mapping) / np.sum((mat3 == i))
                    for i in range(ntarget)]
    colors = map(cmap, centroids / np.max(mapping * 1.0))
    xvals = np.arange(np.max(mapping) + 1) * res / 1.0e6
    yvals1 = np.ones_like(xvals, dtype='float64') * np.nan
    yvals1[mapping] = qp[0] + shift
    for i in range(ntarget):
        yvals1 = np.ones_like(xvals, dtype='float64') * np.nan
        yvals2 = np.ones_like(xvals, dtype='float64') * np.nan
        yvals1[mapping] = qp[i] + shift
        yvals2[mapping] = qp[i + 1] + shift
        ax.fill_between(xvals, yvals1, yvals2, facecolor=colors[i],
                        where=(yvals2 > yvals1), lw=0)
        thisblock = mat2[i]
        thisblockedgeL = np.nonzero((thisblock[:-1] - thisblock[1:]) < 0)[0]
        for edgeL in thisblockedgeL:
            if mat3[edgeL] >= 0 and mat3[edgeL + 1] >= 0:
                xvs = [mapping[edgeL + 1] * res / 1.0e6] * 2
                yvs = [shift, shift + 1]
                ax.plot(xvs, yvs, 'k')
    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(xvals), np.max(xvals))


def _plot_lab_points(lab, tset, ax, pars, mappingdata, cytobanddata,
                scale='pow', n=1.0, cmap=None, s=None):
    """
    Plot effective inter-target Laplacian.
    Mode: Represent by points.
    """
    # Unpack pars
    res = pars['res']
    mapping, nbins = mappingdata
    if s is None:
        s = 20
    # Grid lines
    factor = res / 1.0e6
    minpos = 0.0
    maxpos = nbins * factor
    bandnum = np.zeros(nbins, dtype=int)
    for i, blims in enumerate(cytobanddata[1]):
        ia, ib = map(int, blims)
        bandnum[ia:ib + 1] = i
    centmask = (np.array(cytobanddata[3])[bandnum] == 'acen')
    centpos = np.nonzero(centmask)[0] * factor
    mincent = np.min(centpos)
    maxcent = np.max(centpos) + factor
    xvalss = [[minpos, mincent], [maxcent, maxpos]]
    bandbounds = bandnum[1:] - bandnum[:-1]
    for bpos in np.nonzero(bandbounds)[0]:
        if cytobanddata[0][bpos] == -1 or \
                cytobanddata[0][bpos + 1] == -1:
                    continue
        bposp = bpos + 1
        yvals = [bposp * factor] * 2
        for i in range(2):
            ax.plot(xvalss[i], yvals, '0.3', lw=0.5)
            ax.plot(yvals, xvalss[i], '0.3', lw=0.5)
    maskval = 0.5
    bgmask = np.zeros((nbins, nbins)) + 0.3
    bgmask[centmask] = maskval
    bgmask[:, centmask] = maskval
    _plot_heatmap(bgmask, ax, res, cmap='binary', vlims=[0.0, 1.0])
    # kab proper
    cmap = plt.get_cmap('jet')
    labrescale = (lab - np.min(lab[lab > 0.0])) / (np.max(lab) -
                    np.min(lab[lab > 0.0]))
    if scale == 'pow':
        labrescale = labrescale ** n
    elif scale == 'log':
        labrescale = np.log(labrescale)
    else:
        print 'Invalid scale!'
    clrarray = cmap(labrescale)
    tsetsort = np.sort(tset)
    for i, t1 in enumerate(tsetsort):
        xval = (mapping[t1] + 0.5) * res / 1.0e6
        for j, t2 in enumerate(tsetsort):
            yval = (mapping[t2] + 0.5) * res / 1.0e6
            if i == j:
                ax.scatter(xval, yval, marker='+', c='k', s=s)
            else:
                ax.scatter(xval, yval, marker='o', c=clrarray[i, j],
                                s=s * 0.6, lw=0, alpha=1.0)
    ax.set_axis_bgcolor((0.8, 0.8, 0.8))
    ax.set_xlim(0, nbins * res / 1.0e6)
    ax.set_ylim(nbins * res / 1.0e6, 0)
    return


def _plot_lab_block(lab, qAi, tset, ax, pars, mappingdata,
                scale='pow', n=1.0, cmap=None, s=None,
                badcolor=(1.0, 1.0, 1.0, 0.0)):
    """
    Plot effective inter-target Laplacian.
    Mode: Represent by blocks with area corresponding to most
    strongly-associated target.
    """
    res = pars['res']
    ntarget, npx = qAi.shape
    mapping, nbins = mappingdata
    if s is None:
        s = 20
    mvals = np.argmax(qAi, axis=0)
    # Version 1
    lij = np.zeros((npx, npx))
    for i in range(npx):
        for j in range(npx):
            if mvals[i] == mvals[j]:
                lij[i, j] = np.nan
            else:
                lij[i, j] = lab[mvals[i], mvals[j]]
    #rowsum = np.sum(kij, axis=0)
    #kijp = kij / np.outer(rowsum, rowsum)
    lijpad = _build_fullarray(lij, mappingdata, np.nan)
    if scale == 'pow':
        lijpad = lijpad ** n
    elif scale == 'log':
        lijpad = np.log(lijpad)
    else:
        print 'Invalid scale!'
    _plot_heatmap(lijpad, ax, res, cmap=cmap, badcolor=badcolor)
    # Mark out targets
    for tgt in tset:
        xval = (mapping[tgt]) * res / 1.0e6
        ax.scatter(xval, xval, marker='+', c='r', s=s)
    ax.set_xlim(0, nbins * res / 1.0e6)
    ax.set_ylim(nbins * res / 1.0e6, 0)


def _plot_lab_block_2(lab, membership, ax, pars, mappingdata,
                scale='pow', n=1.0, cmap=None, s=None,
                badcolor=(1.0, 1.0, 1.0, 0.0), returnimg=False):
    """
    Plot effective inter-target Laplacian.
    Mode: Represent by blocks with area corresponding to most
    strongly-associated target.
    Note: membership function has been padded with empty data rows.
    """
    res = pars['res']
    npart, npx = membership.shape
    mapping, nbins = mappingdata
    if s is None:
        s = 20
    mvals = [(np.argmax(v) if np.sum(v) > 0.0 else -1) for v in membership.T]
    #mvals = np.argmax(membership, axis=0)
    # Version 1
    lij = np.zeros((npx, npx)) + np.nan
    for i in range(npx):
        if mvals[i] == -1:
            continue
        for j in range(npx):
            if mvals[j] == -1:
                continue
            if mvals[i] == mvals[j]:
                lij[i, j] = np.nan
            else:
                lij[i, j] = lab[mvals[i], mvals[j]]
    if scale == 'pow':
        lij = lij ** n
    elif scale == 'log':
        lij = np.log(lij)
    else:
        print 'Invalid scale!'
    img = _plot_heatmap(lij, ax, res, cmap=cmap, badcolor=badcolor,
                    returnimg=returnimg)
    ax.set_xlim(0, nbins * res / 1.0e6)
    ax.set_ylim(nbins * res / 1.0e6, 0)
    if returnimg:
        return lij, img
    else:
        return lij


def _plot_lab_block_3(lab, membership, ax, pars, mappingdata,
                scale='pow', n=1.0, cmap=None, s=None,
                badcolor=(1.0, 1.0, 1.0, 0.0)):
    """
    Plot effective inter-target Laplacian.
    Mode: Represent by blocks with area corresponding to most
    strongly-associated target.
    Note: membership function has been padded with empty data rows.
    """
    res = pars['res']
    npart, npx = membership.shape
    mapping, nbins = mappingdata
    if s is None:
        s = 20
    mvals = np.array([(np.argmax(v) if np.sum(v) > 0.0 else -1)
                    for v in membership.T])
    bounds = np.abs(np.array([-2] + list(mvals)) -
                         np.array(list(mvals) + [-2])) > 0
    bounds = np.nonzero(bounds)[0]
    sizes = bounds[1:] - bounds[:-1]
    npieces = len(sizes)
    lims = [bounds[i:i + 2] for i in range(npieces)]
    pieceids = [int(mvals[l[0]]) for l in lims]
    edges = [0.0] + list(np.cumsum(sizes) * res / 1.0e6)
    labexpand = np.zeros((npieces, npieces))
    for i in range(npieces):
        for j in range(npieces):
            labexpand[i, j] = np.nan if ((pieceids[i] < 0 or pieceids[j] < 0)
                            or (pieceids[i] == pieceids[j])) \
                    else lab[pieceids[i], pieceids[j]]
    #ax.pcolor(edges, edges, labexpand)
    _plot_heatmap_uneven(labexpand, edges, ax, res, cmap=cmap,
                badcolor=badcolor)
    print len(edges), labexpand.shape
    #mvals = np.argmax(membership, axis=0)
    # Version 1
    #lij = np.zeros((npx, npx)) + np.nan
    #for i in range(npx):
        #if mvals[i] == -1:
            #continue
        #for j in range(npx):
            #if mvals[j] == -1:
                #continue
            #if mvals[i] == mvals[j]:
                #lij[i, j] = np.nan
            #else:
                #lij[i, j] = lab[mvals[i], mvals[j]]
    #if scale == 'pow':
        #lij = lij ** n
    #elif scale == 'log':
        #lij = np.log(lij)
    #else:
        #print 'Invalid scale!'
    #_plot_heatmap(lij, ax, res, cmap=cmap, badcolor=badcolor)
    ax.set_xlim(0, nbins * res / 1.0e6)
    ax.set_ylim(nbins * res / 1.0e6, 0)
    return


def _plot_lab_block_2chr(lab, targetmembership, tsets, ax, pars, mappingdatas,
                scale='pow', n=1.0, cmap=None, s=None,
                badcolor=(1.0, 1.0, 1.0, 0.0)):
    """
    Plot effective inter-target Laplacian.
    Mode: Represent by blocks with area corresponding to most
    strongly-associated target.
    """
    res = pars['res']
    nchrs = len(tsets)
    nfullbins = [md[1] for md in mappingdatas]
    ntarget, npx = targetmembership.shape
    if s is None:
        s = 20
    mvals = np.array([np.argmax(v) if np.max(v) > 0.0 else np.nan
                    for v in targetmembership.T])
    # Version 1
    lij = np.ones((npx, npx)) * np.nan
    for i in range(npx):
        for j in range(npx):
            if mvals[i] == mvals[j] or np.isnan(mvals[i]) or np.isnan(mvals[j]):
                lij[i, j] = np.nan
            else:
                lij[i, j] = lab[mvals[i], mvals[j]]
    _plot_heatmap(np.log(lij), ax, res, cmap=cmap, badcolor=badcolor)
    # Divider for chromosomes
    for chrid in range(nchrs - 1):
        boundary = np.sum(nfullbins[:chrid + 1]) * res / 1.0e6
        ax.axvline(x=boundary, c='k')
        ax.axhline(y=boundary, c='k')
    # Mark out targets
    for chrid, ts in enumerate(tsets):
        for tgt in ts:
            xval = (mappingdatas[chrid][0][tgt] +
                            np.sum(nfullbins[:chrid])) * res / 1.0e6
            ax.scatter(xval, xval, marker='+', c='r', s=s)
    # Reset plot boundaries
    size = np.sum(nfullbins) * res / 1.0e6
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)


#####################################################

# Graph plotters


def _draw_interpartition(lab, nodelist, cutoffvals, posnodes, posnodelabels, ax,
                nodesize=700, nodefontsize=20, nodecolors=None,
                edgelevels=None, plotloops=False, nodelabels=None,
                nodeshapes=None, levelgraphlayout=False):
    """
    Draw inter-partition interaction graph.
    Input:
        - lab: (N, N) array of effective interaction values
        - nodelist: (N,) list of node labels
        - cutoffvals: (m,) list of cutoff values determining ranges
                   of edge weights to plot.
        - posnodes: (dict) Positions of nodes
        - posnodelabels: (dict) Positions of node labels
        - ax: Matplotlib axis to plot on
        - nodecolors: (dict) fill colors of nodes, or single color for all.
        - edgelevels: (m+1,) list of kwargs dicts for plotting edges.
                      Last entry is for self-loops
                      Default: Use only 3 levels with pre-set parameters.
        - plotloops: Whether or not to plot self-loops.
    """
    G = nx.Graph()
    edgelabels = {}
    for i, u in enumerate(nodelist):
        for j, v in enumerate(nodelist):
            if i <= j:
                G.add_edge(u, v, weight=lab[i, j])
                edgelabels[u, v] = ('%.1e' % lab[i, j])
    if posnodes is None:
        if levelgraphlayout:
            Gp = nx.Graph()
            for i, u in enumerate(nodelist):
                for j, v in enumerate(nodelist):
                    if i <= j:
                        Gp.add_edge(u, v, weight=np.sum(lab[i, j] > cutoffvals))
            posnodes = nx.spring_layout(Gp)
        else:
            posnodes = nx.spring_layout(G)
        #posnodes = nx.spectral_layout(G)
        posnodelabels = posnodes
    #print posnodes
    if edgelevels is None:
        cutoffs = cutoffvals[:3]
        edgelevels = [
            {'edge_color': 'r', 'width': 6},
            {'edge_color': 'k', 'width': 4, 'alpha': 0.8, 'style': 'dashed'},
            {'edge_color': 'b', 'width': 2, 'alpha': 0.5, 'style': 'dashed'},
            {'edge_color': 'g', 'width': 2}
            ]
    else:
        cutoffs = cutoffvals
    edgegroups = []
    edgegroups.append([(u, v) for (u, v, d) in G.edges(data=True) if
                d['weight'] > cutoffs[0]])
    for i in range(len(cutoffs) - 1):
        edgegroups.append([(u, v) for (u, v, d) in G.edges(data=True) if
                d['weight'] <= cutoffs[i] and d['weight'] > cutoffs[i + 1]])
    edgeself = [(u, v) for (u, v, d) in G.edges(data=True) if
                d['weight'] > cutoffs[-1] and u == v]
    if nodeshapes is None:
        if nodecolors is None:
            nx.draw_networkx_nodes(G, posnodes, node_size=nodesize, ax=ax)
        elif isinstance(nodecolors, basestring):
            nx.draw_networkx_nodes(G, posnodes, node_size=nodesize,
                            node_color=nodecolors, ax=ax)
        else:
            for u, c in nodecolors.iteritems():
                Gp = G.subgraph([u])
                nx.draw_networkx_nodes(Gp, posnodes, node_size=nodesize[nodelist.index(u)], ax=ax,
                                    node_color=c, cmap=cm.bone, vmin=0.0,
                                    vmax=1.0)
    else:
        if nodecolors is None:
            for u, c in posnodes.iteritems():
                Gp = G.subgraph([u])
                shape = nodeshapes[u]
                nx.draw_networkx_nodes(Gp, posnodes, node_size=nodesize, ax=ax,
                                    node_shape=shape)
        else:
            for u, c in nodecolors.iteritems():
                Gp = G.subgraph([u])
                shape = nodeshapes[u]
                nx.draw_networkx_nodes(Gp, posnodes, node_size=nodesize, ax=ax,
                                    node_color=c, cmap=cm.bone, vmin=0.0,
                                    vmax=1.0, node_shape=shape)
    if nodelabels is not None:
        nx.draw_networkx_labels(G, posnodelabels, font_size=nodefontsize,
                        labels=nodelabels)
    for edgegroup, edgelevel in zip(edgegroups, edgelevels):
        #print edgegroup
        nx.draw_networkx_edges(G, posnodes, edgelist=edgegroup, ax=ax,
                        **edgelevel)
    if plotloops:
        nx.draw_networkx_edges(G, posnodes, edgelist=edgeself, ax=ax,
                        draw_fine=True, **edgelevels[-1])


def _draw_interpartition_pie(lab, nodelist, cutoffvals, posnodes,
                posnodelabels, ax, nodepievals, nodepiekeylist,
                nodesize=700, nodefontsize=20, basecolor='k',
                edgelevels=None, plotloops=False, nodelabels=None):
    """
    Draw inter-partition interaction graph.
    Input:
        - lab: (N, N) array of effective interaction values
        - nodelist: (N,) list of node labels
        - cutoffvals: (m,) list of cutoff values determining ranges
                   of edge weights to plot.
        - posnodes: (dict) Positions of nodes
        - posnodelabels: (dict) Positions of node labels
        - ax: Matplotlib axis to plot on
        - nodecolors: (dict) fill colors of nodes, or single color for all.
        - edgelevels: (m+1,) list of kwargs dicts for plotting edges.
                      Last entry is for self-loops
                      Default: Use only 3 levels with pre-set parameters.
        - plotloops: Whether or not to plot self-loops.
    """
    G = nx.Graph()
    edgelabels = {}
    for i, u in enumerate(nodelist):
        for j, v in enumerate(nodelist):
            if i <= j:
                G.add_edge(u, v, weight=lab[i, j])
                edgelabels[u, v] = ('%.1e' % lab[i, j])
    if posnodes is None:
        posnodes = nx.spring_layout(G)
        #posnodes = nx.spectral_layout(G)
        posnodelabels = posnodes
    #print posnodes
    if edgelevels is None:
        cutoffs = cutoffvals[:3]
        edgelevels = [
            {'edge_color': 'r', 'width': 6},
            {'edge_color': 'k', 'width': 4, 'alpha': 0.8, 'style': 'dashed'},
            {'edge_color': 'b', 'width': 2, 'alpha': 0.5, 'style': 'dashed'},
            {'edge_color': 'g', 'width': 2}
            ]
    else:
        cutoffs = cutoffvals
    edgegroups = []
    edgegroups.append([(u, v) for (u, v, d) in G.edges(data=True) if
                d['weight'] > cutoffs[0]])
    for i in range(len(cutoffs) - 1):
        edgegroups.append([(u, v) for (u, v, d) in G.edges(data=True) if
                d['weight'] <= cutoffs[i] and d['weight'] > cutoffs[i + 1]])
    edgeself = [(u, v) for (u, v, d) in G.edges(data=True) if
                d['weight'] > cutoffs[-1] and u == v]
    for u in nodelist:
        pos = posnodes[u]
        thissize = nodesize[nodelist.index(u)]
        thisvals = [nodepievals[u][s][0] for s in nodepiekeylist]
        thiscolors = [nodepievals[u][s][1] for s in nodepiekeylist]
        #Gp = G.subgraph([u])
        #nx.draw_networkx_nodes(Gp, posnodes, node_size=thissize * 1.05, ax=ax,
                            #node_color='w', cmap=cm.bone, vmin=0.0,
                            #vmax=1.0)
        w, t = ax.pie(thisvals, colors=thiscolors, startangle=90,
                        counterclock=False, center=pos, radius=thissize)
        for wg in w:
            wg.set_linewidth(0)
        ecangle = 360.0 * (1.0 - float(thisvals[0]) / np.sum(thisvals))
        c = matplotlib.patches.Arc(pos, 2 * thissize, 2 * thissize, angle=90.0,
                        theta1=ecangle, theta2=360.0,
                        edgecolor=basecolor, linewidth=0.5)
        ax.add_patch(c)
    if nodelabels is not None:
        nx.draw_networkx_labels(G, posnodelabels, font_size=nodefontsize,
                        labels=nodelabels)
    for edgegroup, edgelevel in zip(edgegroups, edgelevels):
        #print edgegroup
        nx.draw_networkx_edges(G, posnodes, edgelist=edgegroup, ax=ax,
                        **edgelevel)
    if plotloops:
        nx.draw_networkx_edges(G, posnodes, edgelist=edgeself, ax=ax,
                        draw_fine=True, **edgelevels[-1])


def _draw_interpartition_axes(lab, nodelist, cutoffvals, posnodes,
                fig, ax, axissizes=[0.1, 0.05], nodefontsize=20,
                edgelevels=None, nodelabels=None):
    """
    Draw inter-partition interaction graph: Put pyplot axes in place of nodes.
    Input:
        - lab: (N, N) array of effective interaction values
        - nodelist: (N,) list of node labels
        - cutoffvals: (m,) list of cutoff values determining ranges
                   of edge weights to plot.
        - posnodes: (dict) Positions of nodes
        - posnodelabels: (dict) Positions of node labels
        - ax: Matplotlib axis to plot on
        - edgelevels: (m+1,) list of kwargs dicts for plotting edges.
                      Last entry is for self-loops
                      Default: Use only 3 levels with pre-set parameters.
        - plotloops: Whether or not to plot self-loops.
    """
    G = nx.Graph()
    edgelabels = {}
    for i, u in enumerate(nodelist):
        for j, v in enumerate(nodelist):
            if i <= j:
                G.add_edge(u, v, weight=lab[i, j])
                edgelabels[u, v] = ('%.1e' % lab[i, j])
    if posnodes is None:
        posnodes = nx.spring_layout(G)
        #posnodes = nx.spectral_layout(G)
    #print posnodes
    if edgelevels is None:
        cutoffs = cutoffvals[:3]
        edgelevels = [
            {'edge_color': 'r', 'width': 6},
            {'edge_color': 'k', 'width': 4, 'alpha': 0.8, 'style': 'dashed'},
            {'edge_color': 'b', 'width': 2, 'alpha': 0.5, 'style': 'dashed'},
            {'edge_color': 'g', 'width': 2}
            ]
    else:
        cutoffs = cutoffvals
    edgegroups = []
    edgegroups.append([(u, v) for (u, v, d) in G.edges(data=True) if
                d['weight'] > cutoffs[0]])
    for i in range(len(cutoffs) - 1):
        edgegroups.append([(u, v) for (u, v, d) in G.edges(data=True) if
                d['weight'] <= cutoffs[i] and d['weight'] > cutoffs[i + 1]])
    # Draw axes
    ## Limit coordinate range, re-center coordinates
    xsize, ysize = axissizes
    xspan = 0.8 - xsize
    yspan = 0.8 - ysize
    xmin = posnodes[nodelist[0]][0]
    xmax = posnodes[nodelist[0]][0]
    ymin = posnodes[nodelist[0]][1]
    ymax = posnodes[nodelist[0]][1]
    for n in nodelist:
        xmin = min(xmin, posnodes[n][0])
        xmax = max(xmax, posnodes[n][0])
        ymin = min(ymin, posnodes[n][1])
        ymax = max(ymax, posnodes[n][1])
    dx = xmax - xmin
    dy = ymax - ymin
    d = max(dx / xspan, dy / yspan)
    axdict = {}
    for n in nodelist:
        posnodes[n] = np.array(posnodes[n]) / d
    xmin = posnodes[nodelist[0]][0]
    ymin = posnodes[nodelist[0]][1]
    for n in nodelist:
        xmin = min(xmin, posnodes[n][0])
        ymin = min(ymin, posnodes[n][1])
    ## Create inset axes
    for n in nodelist:
        x, y = posnodes[n][0] - xmin + 0.1, posnodes[n][1] - ymin + 0.1
        print n, x, y
        axdict[n] = fig.add_axes([x, y, xsize, ysize])
    # Draw node labels
    if nodelabels is not None:
        for n, x in axdict.iteritems():
            x.set_xlabel(nodelabels[n], fontsize=nodefontsize)
    # Draw edges
    for edgegroup, edgelevel in zip(edgegroups, edgelevels):
        nx.draw_networkx_edges(G, posnodes, edgelist=edgegroup, ax=ax,
                        **edgelevel)
    return axdict


#####################################################

# Partition plotters


def _plot_donut(ax, ratios, colors, radius=2.0, width=0.95, edgecolors=None,
                startangle=90, counterclock=False):
    """
    Plot donut chart.
    """
    donutargs = {
        'colors': colors,
        'startangle': startangle,
        'counterclock': counterclock
        }
    w, _ = ax.pie(ratios, radius=radius, **donutargs)
    if edgecolors is None:
        edgecolors = ['k'] * len(w)
    for i, we in enumerate(w):
        plt.setp(we, width=width, edgecolor=edgecolors[i])


def _get_partitionlevelcolor(ntarget, ntmin, ntmax, mode='pow', n=-1.0):
    mincolor = np.array((0.28, 0.47, 0.81))
    maxcolor = np.array((1.0, 1.0, 1.0))
    if ntarget >= ntmax:
        return maxcolor
    if ntarget <= ntmin:
        return mincolor
    if mode == 'pow':
        minval = ntmin ** n
        maxval = ntmax ** n
        thisval = ntarget ** n
        return mincolor + (thisval - minval) * (maxcolor - mincolor) / \
                        (maxval - minval)
    elif mode == 'log':
        minval = np.log(ntmin)
        maxval = np.log(ntmax)
        thisval = np.log(ntarget)
        return mincolor + (thisval - minval) * (maxcolor - mincolor) / \
                        (maxval - minval)
    else:
        return mincolor


def _plot_chromosomePartitions_donut(ax, sizes, pieceids, radius=2.0,
        width=0.95, originangle=90, pqspace=0.1, colors=None, edgecolor=None):
    """
    Automated plotting of partitioning scheme in donut form.
    Insert a space corresponding to fraction pqspace between p- and q-ends.
    """
    # Find ratios corresponding to each partition
    sizes2 = [pqspace] + list(copy.deepcopy(sizes) / float(np.sum(sizes)) *
                    (1.0 - pqspace))
    # Define wedge colors, and wedge edge colors
    if colors is None:
        colors2 = ['none']
        for pid in pieceids:
            colors2.append('none' if pid < 0 else (0.28, 0.47, 0.81))
    else:
        colors2 = ['none'] + colors
    edgecolors = ['none'] + [edgecolor if edgecolor is not None else 'k'] * \
                    len(sizes)
    # Plot
    startangle = originangle + (pqspace * 180.0)
    print startangle
    _plot_donut(ax, sizes2, colors2, radius=radius, width=width,
            edgecolors=edgecolors, startangle=startangle, counterclock=False)


#####################################################
# Wrappers


