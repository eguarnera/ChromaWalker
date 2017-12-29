#!/usr/bin/env python
"""
Collection of file I/O functions and utilities for Hi-C analysis

Part of ChromaWalker package
"""


import os
from time import sleep
import cPickle as pickle


##########################################################################
# Data / archive utilities
## Basic file I/O handling


def _pickle_secureunlock(fname):
    """
    Release lock on a file where free=False upon reading just now,
    requiring no changes.
    """
    fname2 = fname + '-lock'
    if os.path.isfile(fname2):
        os.remove(fname2)


def _pickle_secureread(fname, free=False):
    """
    Read a pickled data structure securely (ensure only 1 thread reading it).
    free: Whether or not to let other processes read the same file after I'm
          done... Set to False to avoid race conditions.
    """
    if not os.path.isfile(fname):
        print '%s is not a file!' % fname
        return
    fname2 = fname + '-lock'
    while True:
        if not os.path.isfile(fname2):
            open(fname2, 'a').close()
            break
        print 'Waiting for file lock to free up...'
        sleep(1)
    data = pickle.load(open(fname, 'rb'))
    if free:
        os.remove(fname2)
    return data


def _pickle_securereads(fnames, free=False):
    """
    Read a pickled data structure securely (ensure only 1 thread reading it).
    free: Whether or not to let other processes read the same file after I'm
          done... Set to False to avoid race conditions.
    Reads list of files, with first file as lock.
    """
    #print 'Read', fnames
    fname = fnames[0]
    if not os.path.isfile(fname):
        print '%s is not a file!' % fname
        return
    fname2 = fname + '-lock'
    while True:
        if not os.path.isfile(fname2):
            open(fname2, 'a').close()
            break
        print 'Waiting for file lock to free up...'
        sleep(1)
    data = []
    for fn in fnames:
        data.append(pickle.load(open(fn, 'rb')))
    if free:
        os.remove(fname2)
    return data


def _pickle_securedump(fname, data, freed=False):
    """
    Write a pickled data structure securely (ensure only 1 thread writing it).
    freed: Whether or not the file access was locked. Set to False to prevent
           race conditions.
    """
    fname2 = fname + '-lock'
    if freed:
        while True:
            if not os.path.isfile(fname2):
                open(fname2, 'a').close()
                break
            print 'Waiting for file lock to free up...'
            sleep(1)
    pickle.dump(data, open(fname, 'wb'))
    if os.path.isfile(fname2):
        os.remove(fname2)
    return


def _pickle_securedumps(fnames, datas, freed=False):
    """
    Write a pickled data structure securely (ensure only 1 thread writing it).
    freed: Whether or not the file access was locked. Set to False to prevent
           race conditions.
    Writes list of files, with first file as lock.
    """
    #print 'Write', fnames
    fname = fnames[0]
    fname2 = fname + '-lock'
    if freed:
        while True:
            if not os.path.isfile(fname2):
                open(fname2, 'a').close()
                break
            print 'Waiting for file lock to free up...'
            sleep(1)
    for fn, dt in zip(fnames, datas):
        pickle.dump(dt, open(fn, 'wb'))
    if os.path.isfile(fname2):
        os.remove(fname2)
    return