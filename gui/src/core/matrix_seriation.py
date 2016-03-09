#!/usr/bin/python

#---------------------------------------------------------
# Name: matrix_seriation.py
# Purpose: Implementation of amtrix seriation using rpy2 and seriation package
# Author:	Jacob Retallick
# Created: 2015.10.06
# Last Modified: 2015.10.07
#---------------------------------------------------------

import numpy as np
import rpy2.rinterface as rint
import rpy2.robjects as robj
from rpy2.robjects.packages import importr


# initialise R
rint.initr()

# load packages
seriation = importr("seriation")
stats = importr("stats")


def mat_to_R(M):
    '''convert a numpy matrix to R format'''
    d = M.reshape([1, -1]).tolist()[0]
    return robj.r.matrix(d, nrow=M.shape[0])


def deres_mat(M, res):
    '''Reduce the numerical resolution of a matrix.
    inputs: res     -> resolution (smallest relative diffence between values)
    '''

    factor = res*np.max(np.abs(M))       # absolute resolution factor
    return np.round(M/factor)*factor


def enum_mat(M, hilo=False):
    '''enumerate the non-zero values of M. Set hilo to True if smaller values
    should get larger enumeration.'''

    # get ordered list of non-zero values
    values = set(M.reshape([1, -1]).tolist()[0])
    pairs = [(abs(v), np.sign(v)) for v in values if v != 0]
    values = [p[1]*p[0] for p in sorted(pairs, reverse=hilo)]

    # create map from values to enums
    key_map = {values[i]: i+1 for i in xrange(len(values))}

    enum = M*0  # empty version of M, zero values set by default

    # load enums
    for val in key_map:
        enum += key_map[val]*(M == val)

    return enum


def seriate(M, method="MDS"):
    '''find optimal seriation of a real matrix'''

    mat = mat_to_R(M)           # matrix in R format
    dist = stats.as_dist(mat)   # dissimilarity matrix format
    o = seriation.seriate(dist, method)    # run seriation

    inds = [ind-1 for ind in list(o[0])]

    # candidate reordering
    temp = M[inds, :][:, inds]

    # check for polarity
    for i in xrange(M.shape[0]):
        m = np.array(temp.diagonal(i)).flatten()
        a = m - m[::-1]
        check = np.any(a)
        if check:  # not symmetric in the diagonal
            p = np.sign(a[np.nonzero(a)[0][0]])
            if p < 0:
                inds.reverse()
            break
    else:
        # matrix is symmetric in all diagonals so polarity is arbitrary
        pass

    return inds
