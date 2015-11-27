#!/usr/bin/env python

#---------------------------------------------------------
# Name: auxil.py
# Purpose: Auxiliary function for use in embedder application
# Author: Jacob Retallick
# Created: 2015.11.26
# Last Modified: 2015.11.26
#---------------------------------------------------------

import numpy as np

# physical parameters
eps0 = 8.85412e-12  # permittivity of free space
epsr = 12.          # relative permittivity
q0 = 1.602e-19      # elementary charge

CELL_FUNCTIONS = {'QCAD_CELL_NORMAL': 0,
                  'QCAD_CELL_INPUT': 1,
                  'QCAD_CELL_OUTPUT': 2,
                  'QCAD_CELL_FIXED': 3}

CELL_MODES = {'QCAD_CELL_MODE_NORMAL': 0,
              'QCAD_CELL_MODE_CROSSOVER': 1,
              'QCAD_CELL_MODE_VERTICAL': 2,
              'QCAD_CELL_MODE_CLUSTER': 3}

R_MAX = 1.8         # maximum interaction range


### GENERAL FUNCTIONS


def pinch(string, pre, post):
    '''selects the string between the first instance of substring (pre) and
    the last instance of substring (post)'''
    return string.partition(pre)[2].rpartition(post)[0]


def gen_pols(n):
    '''Generate all possible polarizations for n cells'''
    if n <= 0:
        return []
    return [tuple(2*int(x)-1 for x in format(i, '#0{0}b'.format(n+2))[2:])
            for i in xrange(pow(2, n))]


### QCA CELL PROCESSING


def getEk(c1, c2, DR=2):
    '''Compute the kink energy using the qdot positions (in nm) of two cells.
    If the cell displacement is greater than DR return False'''

    # check cell-cell range
    dx = c1['x']-c2['x']
    dy = c1['y']-c2['y']
    if dx*dx+dy*dy > DR*DR:
        return False

    qdots_1 = c1['qdots']
    qdots_2 = c2['qdots']

    # compute displacements

    x1 = [qd['x'] for qd in qdots_1]
    y1 = [qd['y'] for qd in qdots_1]

    x2 = [qd['x'] for qd in qdots_2]
    y2 = [qd['y'] for qd in qdots_2]

    X1 = np.array([x1, y1]).T.reshape([4, 1, 2])
    X2 = np.array([x2, y2]).T.reshape([1, 4, 2])

    R = np.sqrt(np.sum(pow(X1-X2, 2), axis=2))

    if np.min(R) == 0:
        print 'qdot overlap detected'
        return 0.

    # QCADesigner orders qdots either CW or CCW so same and diff configurations
    # are always alternating indices.

    Q = np.array([1, -1, 1, -1])    # template for charge arrangement

    Q = np.outer(Q, Q)

    Ek = -1e9*q0*np.sum((Q/R))/(8*np.pi*eps0*epsr)

    return Ek


def comp_E_nn(spacing):
    '''compute the kink energy for two nearest interacting non-rotated cells'''

    A = 0.588672

    E_nn = A/(spacing*epsr)

    return E_nn


def prepare_convert_adj(cells, spacing, J):
    '''Prepares useful variables for converting from the parse_qca J matrix to
    a reduced adjacency form.

    outputs:    Js  : J scaled by the nearest neighbour interaction of two
                     non-rotated cells.
                T   : Array of cell-cell types for each element of J
                        1  -> non-rotated - non-rotated
                        0  -> non-rotated - rotated
                        -1 -> rotated - rotated
                DX  : X displacements in grid-spacings
                DY  : Y displacements in grid-spacings
    '''

    # scale J by the kink energy of two non-rotated adjacent cells
    E_nn = comp_E_nn(spacing)
    Js = np.round(J/E_nn, 4)

    # determine interaction type of each element of J:
    #   1  -> non-rotated - non-rotated
    #   0  -> non-rotated - rotated
    #   -1 -> rotated - rotated

    rot = [cell['rot'] for cell in cells]   # array of rotated flags
    rot = 1*np.array(rot).reshape([-1, 1])

    T = 1-(rot+rot.T)
    #T = T.astype(int)

    # get displacements between each cell

    X = np.array([cell['x'] for cell in cells]).reshape([-1, 1])
    Y = np.array([cell['y'] for cell in cells]).reshape([-1, 1])

    DX = (X.T - X)/spacing
    DY = (Y - Y.T)/spacing

    return Js, T, DX, DY


def convert_to_full_adjacency(cells, spacing, J):
    '''Convert the J matrix from parse_qca to include only full adjacency
    interactions'''
    
    Js, T, DX, DY = prepare_convert_adj(cells, spacing, J)
    
    return J*(np.power(DX,2)+np.power(DY, 2) < R_MAX**2)


def convert_to_lim_adjacency(cells, spacing, J):
    '''Convert the J matrix from parse_qca to include only limited adjacency
    interactions'''

    STRONG_THRESH = 0.5
    WEAK_THRESH = 0.1

    Js, T, DX, DY = prepare_convert_adj(cells, spacing, J)
    Js = Js*(np.power(DX,2)+np.power(DY, 2) < R_MAX**2)

    # count number of strong interactions
    strong = [np.count_nonzero(np.abs(Js[i]) > STRONG_THRESH)
                for i in xrange(len(cells))]
    
    # count number of weak interactions
    
    return J*(Js != 0)