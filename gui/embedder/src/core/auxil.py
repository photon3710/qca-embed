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

R_MAX = 1.8             # maximum interaction range
STRONG_THRESH = 0.3     # threshold for qualifying as a strong interaction
D_ERR = 0.2             # allowed error in DX, DY equality


### GENERAL FUNCTIONS


def pinch(string, pre, post):
    '''selects the string between the first instance of substring (pre) and
    the last instance of substring (post)'''
    return string.partition(pre)[2].rpartition(post)[0]


def zeq(x, y, err):
    '''returns True if x and y differ by at most err'''
    return abs(x-y) < err


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
    E_nn = np.max(np.abs(J))
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
    
    # 
    
    N = len(cells)
    A = np.zeros([len(cells), len(cells)], dtype=int)

    for i in range(N-1):
        for j in range(i+1, N):
            dx, dy = abs(DX[i, j]), abs(DY[i, j])
            # first check for adapter condition
            if zeq(min(dx, dy), 0.5, D_ERR) and zeq(max(dx, dy), 1, D_ERR):
                A[i, j] = 3
            elif T[i, j] == 0:
                continue
            # (1, 0) or (0, 1) interaction
            elif zeq(dx+dy, 1, D_ERR):
                A[i, j] = 1
            # (1, 1) interaction
            elif zeq(dx, 1, D_ERR) and zeq(dy, 1, D_ERR):
                A[i, j] = -1
            elif zeq(min(dx, dy), 0, D_ERR) and zeq(max(dx, dy), 2, D_ERR):
                A[i, j] = 2
            elif zeq(min(dx, dy), 1, D_ERR) and zeq(max(dx, dy), 2, D_ERR):
                A[i, j] = -2

    A += A.T

    return Js, T, A, DX, DY


def identify_inverters(A):
    '''Identify inverter cells'''
    
    # an inverter is a cell with at most one strong interaction and two weak
    # interactions, each having one strong interaction
    invs = {}    
    N = A.shape[0]

    # count number of strong interactions for each cell
    num_strong = [np.count_nonzero(A[i,:] == 1) for i in xrange(N)]

    for i in xrange(N):
        if num_strong[i] <= 1:   # check if an inverter
            adj = [j for j in range(N) if A[i, j] == -1 and num_strong[j] == 1]
            if len(adj) == 2:
                for k in range(N):
                    if A[adj[0], k] == 1 and A[adj[1], k] == 1:
                        break
                else:
                    invs[i] = adj
    
    return invs


def identify_xovers(A):
    '''Identify all rotated crossover cells in a circuit'''
    
    # xover condition: two cells have A=2 and no path of A:1,1
    N = A.shape[0]
    cands = [(i,j) for i in range(N-1) for j in range(i+1, N) if A[i,j] == 2]
    
    xovers = []
    for cand in cands:
        for k in range(N):
            if abs(A[cand[0], k]) == 1 and abs(A[cand[1], k]) == 1:
                break
        else:
            xovers.append(cand)
    
    return xovers


def convert_to_full_adjacency(J, Js, T, A, DX, DY):
    '''Convert the J matrix from parse_qca to include only full adjacency
    interactions'''
    
    xovers = identify_xovers(A)
    N = A.shape[0]

    F = np.ones([N, N])
    
    for i in range(N-1):
        for j in range(i+1, N):
            if (i, j) in xovers or A[i, j] in [1, -1, 3]:
                continue
            else:
                F[i, j] = 0
                F[j, i] = 0
    
    return J*F


def convert_to_lim_adjacency(J, Js, T, A, DX, DY):
    '''Convert the J matrix from parse_qca to include only limited adjacency
    interactions'''

    # start with full adjacency representation
    Js = np.array(convert_to_full_adjacency(J, Js, T, A, DX, DY))
                
    # get inverters and their included diagonal interactions
    invs = identify_inverters(A)
    
    # clear all diagonal interactions for non-inverters
    N = A.shape[0]
    for i in range(N-1):
        for j in range(i+1, N):
            if A[i, j] != -1:
                continue
            elif (i in invs and j in invs[i]) or (j in invs and i in invs[j]):
                continue
            else:
                Js[i, j] = Js[j, i] = 0.
    
    return J*(Js != 0)