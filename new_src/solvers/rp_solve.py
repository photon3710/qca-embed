#!/usr/bin/env python

#---------------------------------------------------------
# Name: rp_solve.py
# Purpose: Recursive Partitioning solver
# Author:	Jacob Retallick
# Created: 2016.04.11
# Last Modified: 2016.04.14
#---------------------------------------------------------

import numpy as np
import scipy.sparse as sp
import networkx as nx
import json
import os
import re

from collections import Iterable
from numbers import Number
from pymetis import part_graph
from bcp import chlebikova
from itertools import combinations
from time import time

from core import hash_problem, state_to_pol
from sparse import solve as exact_solve
from sparse import solve_sparse

from pprint import pprint

# solver parameters
N_PARTS = 2     # number of partitions at each recursive step

N_THRESH = 5            # largest allowed number of nodes for exact solver
MEM_THRESH = 1e5        # maximum mode count product
STATE_THRESH = 0.05     # required amplitude for state contribution
E_RES = 1e-3            # resolution in energy binning
W_POW = 1               # power for weighting nodes in chlebikova bisection


### SOLUTION CACHE FUNCTIONS

def generate_hash_table(direc=None):
    '''If direc exists, generate the hash table from the directory
    file structure. Otherwise, create the given directory.'''

    hash_table = {}

    if os.path.exists(direc):
        # load hash table from file list
        regex = re.compile('^[mp][0-9]+.json')
        fnames = os.listdir(direc)
        keys = [fname for fname in fnames if regex.match(fname)]

        for key in keys:
            hval = (-1 if key[0]=='m' else 1)*int(key[1:].partition('.')[0])
            hash_table[hval]=key
    else:
        os.makedirs(direc)

    return hash_table

def from_cache(cache_dir, hval, K, hp, inds):
    '''Load and convert Es and modes from cache'''

    inv_map = {k: i for i, k in enumerate(inds)}
    ninds = [inv_map[k] for k in range(len(inds))]

    ext = '{0}{1}.json'.format('m' if hval<0 else 'p', abs(hval))
    fname = os.path.join(cache_dir, ext)

    fp = open(fname, 'r')
    data = json.load(fp)
    fp.close()

    Es = data['Es']
    modes = data['modes']

    Es_ = [E*K for E in Es]
    modes_ = []
    for mds in modes:
        m = []
        for md in mds:
            m.append(tuple(hp*np.array(md)[ninds]))
        modes_.append(m)

    return Es_, modes_

def to_cache(Es, modes, cache_dir, hval, K, hp, inds):
    '''Convert Es and modes to standard form and save to cache'''

    ext = '{0}{1}.json'.format('m' if hval<0 else 'p', abs(hval))
    fname = os.path.join(cache_dir, ext)

    Es_ = [E/K for E in Es]
    modes_ = []
    for mds in modes:
        m = []
        for md in mds:
            m.append(tuple(hp*np.array(md)[inds]))
        modes_.append(m)

    fp = open(fname, 'w')
    json.dump({'Es': Es_, 'modes': modes_}, fp)
    fp.close()

    return ext


### RECURSIVE PARTITIONING FUNCTIONS

def run_pymetis(J, nparts):
    ''' '''

#    print('running {0}-way partitioning...'.format(nparts))
    # run pymetis partitioning
    adj_list = nx.to_dict_of_lists(nx.Graph(J))
    adj_list = [adj_list[k] for k in range(len(adj_list))]
    ncuts, labels = part_graph(nparts, adjacency=adj_list)

    # get indices of each partition
    parts = [[] for _ in range(nparts)]
    for i, p in enumerate(labels):
        parts[p].append(i)

    return parts

def run_chlebikova(J):
    ''' '''

#    print('Running chlebikova...')
    G = nx.Graph(J!=0)
    for k in G:
        G.node[k]['w'] = 1./len(G[k])**W_POW
    V1, V2 = chlebikova(G)

    return [sorted(x) for x in [V1, V2]]

# checked
def compute_rp_tree(J, nparts=2, inds=None):
    '''Build the recursive partition tree of the adjacency matrix J. At each
    step, J is split into nparts partitions.'''

    if inds is None:
        inds = range(J.shape[0])

    tree = {'inds': inds, 'children': []}
    if J.shape[0] <= N_THRESH:
        return tree

    if nparts==2:
        parts = run_chlebikova(J)
    else:
        parts = run_pymetis(J, nparts)

    # recursively build children tree
    for part in parts:
        sub_tree = compute_rp_tree(J[part,:][:,part], nparts=nparts, inds=part)
        tree['children'].append(sub_tree)

    return tree

# checked
def check_tree(tree):
    '''Check that the tree is a valid recursive partition tree'''

    # children partition the root
    if tree['children']:
        union = []
        for child in tree['children']:
            union += child['inds']
        if sorted(union) != range(len(tree['inds'])):
            return False

    # children are valid rp trees for their arrays
    for child in tree['children']:
        if not check_tree(child):
            return False
    else:
        return True

# checked
def partition(h, J, tree):
    '''Partition h and J parameters for the children of the current tree node.
    Returns the h and J parameters for each child as well as a dict '''

    h_p = []
    J_p = []
    C_p = {}

    nc = len(tree['children'])  # number of children for current node

    # on-partition parameters
    for child in tree['children']:
        inds = child['inds']
        h_p.append(h[inds])
        J_p.append(J[inds,:][:,inds])

    # inter-partition parameters
    for i, j in combinations(range(nc), 2):
        inds1, inds2 = [tree['children'][x]['inds'] for x in [i,j]]
        C_p[(i,j)] = J[inds1, :][:, inds2]

    return h_p, J_p, C_p


### PARTITION MODE FORMULATION

def rp_state_to_pol(amps, prod_states=None):
    '''convert the amplitudes and product states of a state to a pol list'''

    if prod_states is None:
        return state_to_pol(amps)

    modes = np.array(prod_states)
    amps = amps.reshape([-1, 1])

    return -np.sum((amps*amps)*modes, 0)

def get_prod_states(state, comp=False):
    '''Get spin representation of the product states which contribute to the
    given state'''

    # number of spins
    N = int(np.log2(state.shape[0]))

    # isolate contributing product states
    inds = (np.abs(state) > STATE_THRESH).nonzero()[0]

    # sort by contribution magnitude
    inds = sorted(inds, key=lambda x: np.abs(state)[x], reverse=True)

    if comp:
        prod_states = inds
    else:
        prod_states = []
        for ind in inds:
            bstr = format(ind, '#0%db' % (N+2))[2::]
            ps = tuple(np.array(list(map(int, bstr)))*2-1)
            prod_states.append(ps)

    return prod_states


def proc_exact_solve(e_vals, e_vecs, e_res):
    '''Process output from exact solve. Bin energies and record new modes
    for each bin'''

    n_states = e_vecs.shape[1]

    states = [e_vecs[:, i] for i in range(n_states)]

    # associate energies with product states

    Es = list(e_vals)
    prod_states = [get_prod_states(state) for state in states]

    # bin energies and correct
    Ebin = []
    PSbin = []

    while Es:
        E = Es[0]
        try:
            i = next(i for i, x in enumerate(Es) if x > E+e_res)
        except:
            i = None
        Ebin.append(E)
        ps = set(reduce(lambda x, y: x+y, prod_states[:i]))
        PSbin.append(ps)
        if i is None:
            Es = []
            prod_states = []
        else:
            Es = Es[i:]
            prod_states = prod_states[i:]

    # redefine Egaps and prod_states
    Es = Ebin
    prod_states = PSbin

    return Es, prod_states


def select_modes(ES, MS):
    '''Based on the energy gaps for the solution of each partition, select
    which product state modes are included. Input modes should be formated
    as vectors of +- 1. MS[p][i] is a list of modes for the i^th energy bin
    of partition p.'''

    Np = len(ES)    # number of partitions

    i_p = [1]*Np     # number of included indices for each partition
    m_p = [0]*Np     # number of included modes for each partition

    prod = 1        # number of modes in product space

    # offset energies by ground state energies
    for Es in ES:
        Es -= Es[0]

    # force inclusion of ground state modes
    for p in range(Np):
        m_p[p] += len(MS[p][0])
        prod *= m_p[p]

    # check fill condition
    assert prod < MEM_THRESH, 'Not enough memory for ground state inclusion'

    # determine number of new product states for each partition
    K = [[] for i in range(Np)]
    for i in range(Np):
        ps = set(MS[i][0])
        for j in range(1, len(MS[i])):
            temp = len(ps)
            ps.update(MS[i][j])
            K[i].append((ES[i][j], i, len(ps)-temp))

    # order K values for prioritising mode inclusion
    order = sorted(reduce(lambda x, y: x+y, K))

    # main inclusion loop
    check = [False]*Np   # set True if term tejected
    while prod < MEM_THRESH and not all(check) and order:
        E, p, k = order.pop(0)  # candidate
        # shouldn't add higher energy term if lower energy rejected
        if check[p]:
            continue
        # check if adding new components pushes prod over threshold
        temp = prod
        prod += k*prod/m_p[p]
        if prod >= MEM_THRESH:
            prod = temp
            check[p] = True
        else:
            i_p[p] += 1
            m_p[p] += k

    # format the modes for output: list of mode matrices

    modes = []

    for p in range(Np):
        mds = []
        for i in range(i_p[p]):
            mds += MS[p][i]
        mds = sorted(set(mds))
        modes.append(np.matrix(mds))

    return modes

def comp_on_partition(h, J, gam, modes):
    '''Compute on-partition parameters'''

    N = h.size

    modes = np.matrix(modes)

    J_ = np.matrix(np.triu(J))
    sz = np.dot(h, modes.T).reshape([-1,])
    sz2 = np.diag(modes*J_*modes.T)

    if gam is None:
        g = None
    else:
        mds = np.asarray(modes)
        diff = np.abs(mds.reshape([-1, 1, N])-mds.reshape([1, -1, N]))
        g = gam*(np.sum(diff, axis=2) == 2)

    return [sz, sz2, g]

def comp_off_partition(C, mds1, mds2):
    '''Compute the inter-partition coupling parameters'''

    JJ = mds1*C*mds2.T

    return JJ

def gen_comp_diag(sz, sz2, J_m):
    '''Compute the diagonal elements of the component Hamiltonian'''

    N = len(sz)     # number of partitions

    # generate size counters
    M = [sz_.shape[1] for sz_ in sz]  # number of modes per partition
    C = np.cumprod(M)   # cumulative product of mode sizes
    C = np.insert(C, 0, 1)  # useful for avoiding special case

    # on comp energy terms
    on_site = np.zeros([1, C[-1]], dtype=float)
    for p in range(N):
        a = sz[p] + sz2[p]
        temp = np.tile(np.repeat(a, C[p]), C[-1]/C[p+1])
        on_site += temp

    # external comp-comp mode coupling
    ext_coup = np.zeros([1, C[-1]], dtype=float)
    for p1, p2 in combinations(range(N), 2):
        # handle p1 behaviour
        j = J_m[(p1, p2)].T
        if not j.any():
            continue
        #j = np.arange(j.size).reshape(j.shape)
        a = np.tile(np.repeat(j, C[p1], axis=1), C[p2]/C[p1+1])
        # handle p2 behaviour
        b = np.tile(a.flatten(), C[-1]/C[p2+1])
        ext_coup += b

    return on_site + ext_coup

def gen_comp_tunn(G):
    '''Compute the off-diagonal tunneling terms in the component formalism. G
    is a list of the on-partition tunneling operators'''

    N = len(G)  # number of partitions
    Nm = [x.shape[0] for x in G]    # number of modes per partition
    Cm = np.insert(np.cumprod(Nm), 0, 1)     # size of each partition sub-block

    # for each mode update the 'diagonal' submatrix
    mat = sp.coo_matrix((1, 1), dtype=float)
    for p in range(N):
        # construct the sub-block container
        A = [[None]*m + [mat] for m in range(Nm[p])]
        for m, n in combinations(range(Nm[p]), 2):
            if G[p][m, n] == 0:
                A[m].append(None)
            else:
                dat = np.repeat(-G[p][m, n], Cm[p])
                A[m].append(sp.diags([dat], [0]))
        mat = sp.bmat(A)

    return mat.T

def build_comp_H(h_p, J_p, C_p, gam, modes):
    '''Build a sparse representation fo the component space Hamiltonian'''

    N = len(h_p)    # number of partitions
    # get on partition parameters
    temp = [comp_on_partition(h, J, gam, m) for h, J, m in zip(h_p, J_p, modes)]
    sz, sz2, g = [list(x) for x in zip(*temp)]

    # get inter-partition parameters
    J_m = {}
    for i, j in combinations(range(N), 2):
        J_m[(i,j)] = comp_off_partition(C_p[(i,j)], modes[i], modes[j])

    # construct diagonal elements
    diag = gen_comp_diag(sz, sz2, J_m).reshape([-1,])

    ### FOR NOW ALLOW PURELY DIAG MATRIX
    diag = sp.diags([diag], [0])

    if gam is None or all(x is None for x in g):
        return diag

    # construct tunneling elements
    off_diag = gen_comp_tunn(g)
    return diag + off_diag + off_diag.T

def general_decomp(n, c):
    '''Decompose a number into a general basis cumprod c'''

    rep = []
    for i in range(len(c)):
        t, n = divmod(n, c[i])
        rep.append(t)

    return rep

def correct_prod_state(pstates, modes, tree):
    '''Correct product state to account for mode space representation'''

    inds = [child['inds'] for child in tree['children']]

    Nps = len(pstates)  # number of product state lists
    N = len(modes)      # number of partitions

    inds = reduce(lambda x, y: x+y, inds)

    nmodes = [mds.shape[0] for mds in modes]    # number of modes per part
    C = np.cumprod([1]+nmodes[:-1])[::-1]     # cumprod of mode counts

    ps_modes = []
    # for each product state list
    for i in range(Nps):
        ps_list = pstates[i]
        ps_mds = []
        for ps in ps_list:
            rep = general_decomp(ps, C)[::-1]
            ps_m = [modes[j][rep[j]].tolist()[0] for j in range(N)]
            ps_m = reduce(lambda x, y: x+y, ps_m)
            # reorder using indices
            ps_mds.append(tuple([x[1] for x in sorted(zip(inds, ps_m))]))
        ps_modes.append(ps_mds)

    return ps_modes

def proc_comp_solve(e_vals, e_vecs, modes, tree, e_res):
    '''Process the output of the sparse solver for the component formalism'''

    ns = e_vecs.shape[1]    # number of states

    Es = list(e_vals)
    states = [e_vecs[:, i] for i in range(ns)]
    pstates = [get_prod_states(state, comp=True) for state in states]

    # bin energies and correct
    Ebin = []
    PSbin = []
    while Es:
        E = Es[0]
        try:
            i = next(i for i, x in enumerate(Es) if x > E+e_res)
        except:
            i = None
        Ebin.append(E)
        ps = set(reduce(lambda x, y: x+y, pstates[:i]))
        PSbin.append(ps)
        if i is None:
            Es = []
            pstates = []
        else:
            Es = Es[i:]
            pstates = pstates[i:]

    # redefine Egaps and prod_states
    Es = Ebin
    pstates = PSbin

    # correct states to account for mode space
    pstates = correct_prod_state(pstates, modes, tree)

    return Es, pstates


def solve_comp(h_p, J_p, C_p, gam, modes, tree, e_res, **kwargs):
    '''Formulate and solve the component Hamiltonian at the current tree node'''

    verbose = kwargs['verbose']
    t = time()

    if verbose:
        print('\nRunning components solver...')

    Hs = build_comp_H(h_p, J_p, C_p, gam, modes)

    # print(Hs.diagonal())

    t1 = time()
    # run sparse matric solver
    if verbose:
        print('H matrix size {0}'.format(str(Hs.shape)))
        print('Running sparse solver...'),

    e_vals, e_vecs = solve_sparse(Hs, more=False)
    if verbose:
        print('solver time: {0:.4f} sec'.format(time()-t1))

    Es, modes = proc_comp_solve(e_vals, e_vecs, modes, tree, e_res)

    if verbose:
        print('Component solver time: {0:.4f}'.format(time()-t))

    try:
        if kwargs['full_output']:
            return Es, modes, e_vals, e_vecs
        raise KeyError
    except KeyError:
        return Es, modes



def recursive_solver(h, J, gam, tree, **kwargs):
    '''Recursive method for identifying energy bins and significant modes'''

    # pull parameters from kwargs
    cache_dir = kwargs['cache_dir']
    verbose = kwargs['verbose']

    # energy resolution
    e_res = np.max(np.abs(J))*E_RES

    if verbose:
        print('Detected problem size: {0}'.format(h.size))

    # try to read from cache
    if cache_dir is not None:
        # get or create hash table
        if 'hash_table' not in kwargs:
            kwargs['hash_table'] = generate_hash_table(direc=cache_dir)
        hval, K, hp, inds = hash_problem(h, J, gam=gam)
        hash_pars = {'hval':hval, 'K':K, 'hp':hp, 'inds':inds}
        # try to look up solution
        if hval in kwargs['hash_table']:
            try:
                Es, modes = from_cache(cache_dir, **hash_pars)
                return Es, modes
            except:
                print('Something went wrong reading from cache')

    # solution not cached, compute
    if not tree['children']:
        if verbose:
            print('Running exact solver...')
        e_vals, e_vecs = exact_solve(h, J, gamma=gam, k=10*len(h))
        Es, modes = proc_exact_solve(e_vals, e_vecs, e_res)
    else:
        if verbose:
            print('Running recursive partitioning...')
        h_p, J_p, C_p = partition(h, J, tree)

        # solve each partition recursively
        ES, MS = [], []
        for h_, J_, tree_ in zip(h_p, J_p, tree['children']):
            Es_, modes_ = recursive_solver(h_, J_, gam, tree_, **kwargs)
            ES.append(np.array(Es_))
            MS.append(modes_)

        # select modes to include
        modes = select_modes(ES, MS)

        # solve component system
        Es, modes = solve_comp(h_p, J_p, C_p, gam, modes, tree, e_res, **kwargs)

    # save to cache
    if 'hash_table' in kwargs:
        try:
            kwargs['hash_table'][hash_pars['hval']] = \
                to_cache(Es, modes, cache_dir, **hash_pars)
        except:
            print('Failed to cache solution...')
    return Es, modes

def out_handler(h, J, gam, prod_states):
    '''Make an estimation of low energy spectrum using the determined
    applicable subset of product states and the problem parameters'''


    # create a single list of the product states
    prod_states = [list(ps) for ps in prod_states]
    pstates = sorted(set(reduce(lambda x, y: x+y, prod_states)))

    # find the energy associated with each product state
    modes = np.matrix(pstates)
    Hs = build_comp_H([h], [J], [], gam, [modes])
    Eps = Hs.diagonal()

    # sort product states by energy
    ps_order = sorted(list(enumerate(Eps)), key=lambda x: x[1])
    ps_order, Eps = zip(*ps_order)

    # find the eigenstates in the reduced mode space
    e_vals, e_vecs = solve_sparse(Hs, more=True)

    E = e_vals
    states = e_vecs
    states = [states[ps_order, i] for i in xrange(len(E))]

    # get cell polarizations for each eigenstate
    state_pols = [rp_state_to_pol(state, pstates) for state in states]

    return E, states, Eps, pstates, state_pols

def rp_solve(h, J, gam=None, **kwargs):
    '''Solve transverse field ising spin-glass configuration using recursive
    partitioning with low-energy spectrum mode composition.

    inputs: h       : iterable of bias parameters
            J       : array of interaction parameters
            gam     : optional, scalar tunneling parameter for all cells

    optional kwargs:
            tree        : pre-computed recursion tree
            cache_dir   : directory for cached solutions
            hash_table  : table of hash values for cached solutions
            verbose     : flag, echo activity.

    outputs:    e_vals  : estimates of lowest eigen-values.
                e_vecs  : estimates of corresponding eigen-states in the basis
                          of modes. (e_vecs[:,i] is the i^th state)
                modes   : array of basis modes. (modes[:,i] is the i^th mode)
    '''

    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    else:
        verbose=False

    if 'cache_dir' in kwargs:
        cache_dir = kwargs['cache_dir']
    else:
        cache_dir = None

    # regularise input formating
    if verbose:
        print('Standardizing input format...')

    assert isinstance(h, Iterable), 'h is not iterable'
    h = np.array(h).reshape([-1,])

    assert isinstance(J, Iterable), 'J is not iterable'
    J = np.array(J)
    assert J.shape == (len(h), len(h)), 'J is not square'

    if isinstance(gam, Number):
        gam = float(gam)
    elif gam is not None:
        raise AssertionError('Invalid gamma format. Must be scalar or None')

    # force J symmetric
    if np.any(np.tril(J,-1)):
        J = J.T
    J = np.triu(J)+np.triu(J, 1).T

    # initialise recursion tree
    if verbose:
        print('Getting recursion tree...')
    try:
        tree = kwargs['tree']
        assert check_tree(tree, ar=range(h)), '\tgiven tree invalid...'
    except KeyError:
        if verbose:
            print('\trunning graph clustering...')
        tree = compute_rp_tree(J != 0, nparts=N_PARTS)

    # run recursive solver
    Es, modes = recursive_solver(h, J, gam, tree, verbose=verbose, cache_dir=cache_dir)

    # estimate eigenvalues and eigenstates
    e_vals, e_vecs, Eps, modes, pols  = out_handler(h, J, gam, modes)

    return e_vals, e_vecs, modes

if __name__ == '__main__':
    pass
