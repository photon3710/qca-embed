#!/usr/bin/python

#---------------------------------------------------------
# Name: auxil.py
# Purpose: Auxiliary commonly used functions
# Author:	Jacob Retallick
# Created: 09.06.2014
# Last Modified: 09.06.2015
#---------------------------------------------------------

import numpy as np
import scipy.sparse as sp
import sys

import networkx as nx
import matplotlib.pyplot as plt
#from itertools import permutations
from numbers import Number      # for scalar instance checking
from collections import Iterable    # iterable check
from matrix_seriation import seriate

## PHYSICAL PARAMETERS
eps0 = 8.85412e-12  # permittivity of free space
epsr = 12.          # relative permittivity
q0 = 1.602e-19      # elementary charge

## QCADESIGNER PARSING PARAMETERS

CELL_FUNCTIONS = {'QCAD_CELL_NORMAL': 0,
                  'QCAD_CELL_INPUT': 1,
                  'QCAD_CELL_OUTPUT': 2,
                  'QCAD_CELL_FIXED': 3}

CELL_MODES = {'QCAD_CELL_MODE_NORMAL': 0,
              'QCAD_CELL_MODE_CROSSOVER': 1,
              'QCAD_CELL_MODE_VERTICAL': 2,
              'QCAD_CELL_MODE_CLUSTER': 3}

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
        sys.exit()

    # QCADesigner orders qdots either CW or CCW so same and diff configurations
    # are always alternating indices.

    Q = np.array([1, -1, 1, -1])    # template for charge arrangement

    Q = np.outer(Q, Q)

    Ek = -1e9*q0*np.sum((Q/R))/(8*np.pi*eps0*epsr)

    return Ek


def comp_E_nn(spacing, OLD_QCAD=False):
    '''compute the kink energy for two nearest interacting non-rotated cells'''

    A = 0.588672
    if OLD_QCAD:
        A = 0.3827320944

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

    R_MAX = 2
    Js, T, DX, DY = prepare_convert_adj(cells, spacing, J)

    xovers = [i for i in range(len(cells)) if is_xover(cells, DX, DY, i)]
    for i in range(len(cells)):
        # check to see if the cell is involved in a cross over
        for j in range(len(cells)):
            # if it is not a cross over and futher than 2 away, strip J
            if not (i in xovers and j in xovers):
                if (DX[i][j]**2 + DY[i][j]**2 >= R_MAX**2):
                        J[i][j] = 0
                        J[j][i] = 0

    return J


def convert_to_lim_adjacency(cells, spacing, J):
    '''Convert the J matrix from parse_qca to include only limited adjacency
    interactions'''

    c_index = range(len(cells))
    R_MAX = 2
    Js, T, DX, DY = prepare_convert_adj(cells, spacing, J)

    xovers = [i for i in c_index if is_xover(cells, DX, DY, i)]
    invs = [i for i in c_index if is_inv(Js, DX, DY, i)]
    print invs

    for i in c_index:
        # number of strong interactions of current cell
        si = len([j for j in c_index if Js[i][j] == 1 or Js[i][j] == -1.472])
        for j in c_index:

            dx = DX[i][j]
            dy = DY[i][j]

            if not (i in xovers and j in xovers):
                if (i in invs or j in invs) and si < 2:
                    if dx**2 + dy**2 >= R_MAX**2:
                        print 'it happened'
                        J[i][j] = 0
                        J[j][i] = 0

                elif dx**2 + dy**2 >= R_MAX:
                        J[i][j] = 0
                        J[j][i] = 0

    return J


def is_xover(cells, DX, DY, i):
    '''check to see if a cell is involved in a cross over'''

    #find cells directly adjacent horizontally
    hor = [j for j in range(len(DY[i])) if DY[i][j] == 0]
    x_adj = [j for j in hor if abs(DX[i][j]) == 1]

    #find cells directly adjacent vertically
    ver = [j for j in range(len(DX[i])) if DX[i][j] == 0]
    y_adj = [j for j in ver if abs(DY[i][j]) == 1]

    #if the pairs of cells are different, than there is a cross over
    if len(x_adj) == 2:
        if cells[x_adj[0]]['rot'] != cells[x_adj[1]]['rot']:
            return True

    if len(y_adj) == 2:
        if cells[y_adj[0]]['rot'] != cells[y_adj[1]]['rot']:
            return True

    # error message if there is more than 2 adjacent cells in either dir
    if len(x_adj) > 2 or len(y_adj) > 2:
        print 'Error: there are %d cells horizontally adjacent' +\
            ' and %d cells vertically adjacent' % (len(x_adj), len(y_adj))

    return False


def is_inv(Js, DX, DY, i):
    '''check to see if a cell is an inverter cell
    and inverter cell is the cell that has two diagonal interactions, and one
    directly adjacent interaction (labelled IC below)
        c - c
        |    \
    c - c     IC - c
        |    /
        c - c
    '''

    index = range(len(Js[i]))
    # find number of strong and medium bonds
    m = [j for j in index if Js[i][j] == -0.2174 or Js[i][j] == 0.172]
    s = [j for j in index if Js[i][j] == 1 or Js[i][j] == -1.472]

    if len(m) >= 2 and len(s) == 1:
        # in case of weird cases - checks to see that strong interactions
        # are on the opposite side of the medium interactions
        opp = 0
        for j in m:
            if DX[i][j] == (-1) * DX[i][s[0]]:
                opp += 1
            if DY[i][j] == (-1) * DY[i][s[0]]:
                opp += 1
        return opp == 2

    return False


def construct_zone_graph(cells, zones, J, feedback, show=False):
    '''Construct a DiGraph for all the zones with keys given by (n, m) where
    n is the shell index and m is the zone index within the shell'''

    # create nodes
    G = nx.DiGraph()
    for i_shell in xrange(len(zones)):
        for i_zones in xrange(len(zones[i_shell])):
            key = (i_shell, i_zones)
            kwargs = {'inds': [], 'fixed': [], 'drivers': [], 'outputs': []}

            for ind in zones[i_shell][i_zones]:
                if cells[ind]['cf'] == CELL_FUNCTIONS['QCAD_CELL_INPUT']:
                    kwargs['drivers'].append(ind)
                elif cells[ind]['cf'] == CELL_FUNCTIONS['QCAD_CELL_FIXED']:
                    kwargs['fixed'].append(ind)
                else:
                    kwargs['inds'].append(ind)
                    if cells[ind]['cf'] == CELL_FUNCTIONS['QCAD_CELL_OUTPUT']:
                        kwargs['outputs'].append(ind)

            G.add_node(key, **kwargs)

    # edges
    for shell in xrange(1, len(zones)):
        for i in xrange(len(zones[shell-1])):
            k1 = (shell-1, i)
            for j in xrange(len(zones[shell])):
                k2 = (shell, j)
                if np.any(J[G.node[k1]['inds'], :][:, G.node[k2]['inds']]):
                    G.add_edge(k1, k2)
                if k2 in feedback:
                    for fb in feedback[k2]:
                        G.add_edge(k2, fb)

    if show:
        plt.figure('Zone-Graph')
        nx.draw_graphviz(G)
        plt.show()

    return G


### HAMILTONIAN GENERATION

PAULI = {}
PAULI['x'] = sp.dia_matrix([[0, 1], [1, 0]])
#PAULI['y'] = sp.dia_matrix([[0, 1j], [-1j, 0]])
PAULI['z'] = sp.dia_matrix([[-1, 0], [0, 1]])


def stateToPol(state):
    '''converts a 2^N size state vector to an N size polarization vector'''

    state = np.asmatrix(state)

    ## correct state alignment

    a, b = state.shape
    if a < b:
        state = state.transpose()

    amp = np.abs(np.multiply(state.conjugate(), state))

    N = int(np.log2(np.size(state)))
    SZ = [np.asmatrix(pauli(i+1, N, 'z')) for i in xrange(N)]

    POL = [-float(SZ[i]*amp) for i in xrange(N)]
    return POL


def pauli(index, N, typ):
    '''computes the tensor product sigma_typ(i) '''

    if index < 1 or index > N:
        print 'Invalid tensor product index...must be in range (1...N)'
        sys.exit()

    if not typ in PAULI.keys():
        print "Invalid pauli matrix type... must be in ['x', 'y', 'z']"
        sys.exit()

    p_mat = PAULI[typ]

    if index == 1:
        product = sp.kron(p_mat, sp.eye(pow(2, N-index)))
    else:
        temp = sp.kron(sp.eye(pow(2, index-1)), p_mat)
        product = sp.kron(temp, sp.eye(pow(2, N-index)))

    if typ == 'z':
        return product.diagonal()
    else:
        return sp.tril(product)


def generateHam(h, J, gamma=None):
    ''' computes the Hamiltonian as a sparse matrix.
    inputs:    h - iterable of size N containing on site energies
            J - matrix (type J[i,j]) containing coupling strengths. Needs
                to contain at least the upper triangular values
    '''

    # handle h format
    h = np.asarray(np.reshape(h, [1, -1])).tolist()[0]

    N = np.size(h)
    N2 = pow(2, N)

    # initialise pauli and data matrices

    #J=sp.triu(J)

    offdiag_flag = False
    if isinstance(gamma, Number):
        offdiag_flag = True
        gamma = [gamma]*N
    elif isinstance(gamma, Iterable):
        offdiag_flag = True
        gamma = np.asarray(np.reshape(gamma, [1, -1])).tolist()[0]

    SZ = [pauli(i+1, N, 'z') for i in xrange(N)]  # diag np.array
    DIAG = np.zeros([1, N2], dtype=float)

    if gamma is not None:
        SX = [pauli(i+1, N, 'x') for i in xrange(N)]  # tril sp mat format
        OFFDIAG = sp.dia_matrix((N2, N2), dtype=float)

    for i in xrange(N):

        DIAG += SZ[i]*h[i]

        for j in xrange(i+1, N):
            DIAG += J[i, j]*np.multiply(SZ[i], SZ[j])

        if offdiag_flag:
            OFFDIAG = OFFDIAG - gamma[i]*SX[i]

    H = sp.diags(DIAG[0], 0)
    if offdiag_flag:
        H = H+OFFDIAG

    upper = sp.tril(H, -1).getH()

    return H+upper


################################################################
## FORMATTING FUNCTIONS


def coefToConn(h, J):
    '''convert the h and J coefficients into a full adjacency list
    for embedding, 0->N indexing '''

    N = len(h)

    D = {i: [] for i in xrange(N)}

    for i in xrange(N):
        d = list(J[i].nonzero()[0])
        for j in d:
            D[i].append(j)
            #D[j].append(i)

    return D


################################################################
### HASHING ALGORITHM


#def get_sub_problems(vals):
#    '''Return a dict of degenerate subproblems'''
#
#    counts = {key: [] for key in set(vals)}
#    for i in xrange(len(vals)):
#        counts[vals[i]].append(i)
#    return {key: counts[key] for key in counts if len(counts[key]) > 1}
#
#
#def sort_by_A(A):
#    ''' '''
#    n = A.shape[0]
#    vals, ninds = [list(x) for x in zip(*sorted(zip(A.tolist(), range(n))))]
#    vals = [tuple(x) for x in vals]
#
#    # get sub-problems of size >1
#    sub_probs = get_sub_problems(vals)
#    sub_probs = [sub_probs[x] for x in sorted(sub_probs)]
#
#    # set flag
#    flag = len(set(vals)) > 1
#
#    return ninds, sub_probs, flag
#
#
#def sort_by_B(B):
#    ''' '''
#
#    n, m = B.shape
#
#    if n > 2:
#        pass
#
#    # generate metrics for each row permutation
#    metrics = {}
#    for perm in permutations(range(n)):
#        metric = np.array(sorted(B[perm, :].T.tolist())).T.tolist()
#        metrics[perm] = metric
#
#    # sort permutations by metric
#    vals, perms = [list(x) for x in
#                   zip(*sorted([(metrics[p], p) for p in metrics]))]
#
#    # isolate lowest metric permutations (all of them)
#    cands = []  # candidate permutations
#    for v, p in zip(vals, perms):
#        if v > vals[0]:
#            break
#        else:
#            cands.append(list(p))
#
#    # pick first as ninds
#    ninds = cands[0]
#
#    return ninds, []


#def sub_seriate(M, pinds, inds):
#    '''Solve the sub-seriation problem in matrix M for problem indices pinds
#    and M seriation indices inds. M and inds are modified in place'''
#
#    m = len(pinds)
#
#    A = M[pinds, 0:pinds[0]]    # pre-matrix
#    B = M[pinds, pinds[-1]+1:]    # post-matrix
#
#    # each sorting approach returns the reordering of pinds, and
#    # the sub-problem indices (with respect to pinds)
#
#    # sort by A
#    if np.any(A):   # skip if A is empty or all zero
#        ninds, sub_probs, flag = sort_by_A(A)
#    else:
#        flag = False
#
#    if not flag and np.any(B):
#        ninds, sub_probs = sort_by_B(B)
#
#    # use ninds to reorder M and update inds
#    ninds = [pinds[i] for i in ninds]
#    M[pinds, :] = M[ninds, :]
#    M[:, pinds] = M[:, ninds]
#    ninds = [inds[i] for i in ninds]
#
#    for i in xrange(m):
#        inds[pinds[i]] = ninds[i]
#
#    # solve each sub-problem of >1 size
#    for sub_prob in sub_probs:
#        if len(sub_prob) == 1:
#            continue
#        spinds = [pinds[i] for i in sub_prob]
#        sub_seriate(M, spinds, inds)


def hash_mat(m):
    '''return a hash value for the given matrix'''
    # disable buffer writeability flag (make immutable)
    m.flags.writeable = False
    # generate hash value
    h = hash(m.data)
    # reet writeable flag
    m.flags.writeable = True
    return h


#def seriate(M, W=None):
#    '''Attempt to obtain a unique seriation of M with on-site immutables W.
#    There may be multiple index permutations which yield the same matrix. Only
#    one such permutation is returned. This method has not yet been robustly
#    check against a large range of circuits'''
#
#    # number of problem nodes
#    N = M.shape[0]
#
#    # map M values to positive integers
#    values = sorted(set(M.ravel().tolist()))
#    values.remove(0)
#
#    J = np.zeros(M.shape, dtype=int)
#
#    for i in xrange(len(values)):
#        J += (M == values[i])*(i+1)
#
#    # compute maximum eigenvalue of G
#    lmax = max(np.linalg.eigvalsh(J != 0))
#
#    # scale factor for integer mapping
#    scale = max(1, int(np.max(np.abs(M))*1e6))
#
#    # Katz centrality
#    alpha = 1.*lmax
#    K = np.linalg.solve(np.eye(N) - alpha*M, np.ones([N, 1]))
#    K = [int(scale*K[i, 0]) for i in range(N)]
#
#    D = np.diag(M)
#
#    if W is None:
#        X = zip(K, D)
#    else:
#        X = zip(W, K, D)
#
#    vals, inds = [list(x) for x in zip(*sorted(zip(X, range(N))))]
#
#    # update J matrix
#    J = J[inds, :][:, inds]
#
#    # determine sub-problems
#    sub_probs = get_sub_problems(vals)
#
#    # solve each sub-problem
#    print sub_probs.values()
#    for key in sorted(sub_probs):
#        sub_seriate(J, sub_probs[key], inds)
#
#    return inds


def formulate_seriation(h, J, g, g0=2):
    '''Incorporate on-site parameters into the Distance matrix for
    seriation'''

    c = .5*(1+h)/(1+g/g0)
    D = np.asmatrix(J+np.diag(c.flatten()))

    DD = np.abs(np.power(D, 2))
    np.fill_diagonal(DD, 0)
    return DD

    vals, vecs = np.linalg.eigh(D)
    alpha = .5/np.max(np.abs(vals))

    S = np.asmatrix(np.diag(alpha*vals/(1-alpha*vals)))
    D = vecs*S*vecs.getH()

    # remove diagonal elements
    np.fill_diagonal(D, 0)

    # force positive distances
    D = np.abs(D)

    return D


def hash_problem(h, J, g):
    '''Obtain a set of candidate hash values for a given (h, J, g) problem

    inputs: h - 1d iterable of on-site energy terms
            J - 2d iterable of coupling terms
            g - 1d iterable of tunneling terms

    outputs: outs - a list of [val, S, s_h, inds] lists.
                    val : hask value
                    S   : absolute scale
                    s_h : polarity of h
                    inds: optimal index permutation
    '''

    # handle input formats
    h = np.array(h).reshape(1, -1)

    N = h.size

    if not hasattr(g, '__len__'):
        g = g*np.ones([1, N], dtype=float)
    else:
        g = np.array(g, dtype=float).reshape(1, -1)

    try:
        J = np.array(J).reshape(N, N)
    except ValueError as e:
        print('J is the wrong size/format...')
        raise e

    assert g.size == N, 'Invalid g size'
    assert np.all(J == J.T), 'J either has diag elements or is not symmetric'

    # extract scale factor and round for tollerance invariance
    S = np.max(np.abs(J))

    assert S > 0, 'No non-zero coupling terms'

    # h, J, and g are unreference copies so can change
    h, J, g = [np.round(x/S, 3) for x in [h, J, g]]

    # sign of h
    s_h = np.sign(np.sum(h))
    s_h = [-1, 1] if s_h == 0 else [s_h]

    outs = []

    for s in s_h:

        # formulate the problem for seriation
        D = formulate_seriation(s*h, J, g)

        # run seriation
        inds = seriate(D, "MDS")
        
#        print D[inds, :][:, inds]

        # reorder matrices
        temp_h = h[0, inds]
        temp_g = g[0, inds]
        temp_J = J[inds, :][:, inds]

        # generate hash key
        key = tuple([hash_mat(m) for m in [temp_h, temp_g, temp_J]])
        val = hash(key)

        outs.append([val, S, s, inds])

    return outs
