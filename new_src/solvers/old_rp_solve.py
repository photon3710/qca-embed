#!/usr/bin/python

from core import state_to_pol
from sparse import solve, solve_sparse
from pymetis import part_graph
from math import ceil
import scipy.sparse as sp
from time import time

# testing methods
from pprint import pprint

import numpy as np
import networkx as nx
import os

## SOLVER PARAMETERS

# threshold values
N_THRESH = 12           # maximum partition size for exact solver
MEM_THRESH = 1e6        # maximum mode count product
STATE_THRESH = 0.02     # required amplitude for state contribution

assert N_THRESH <= 21, 'Given N_THRESH will likely crash the sparse solver...'

# flags and counts
N_PARTS = 2             # maximum number of partitions at each iteration
                        # best results seem to be for 2 partitions
USE_NN = False          # Convert to nearest neighbour

# resolution
E_RES = 5e-2            # resolution for energy binning relative to max J

VERBOSE = False     # verbose flag
CACHE = False       # use cache and hashing
OVERWRITE = False   # force cache overwrite
FULL_ADJ = True     # adjacency type for testing

CACHE_DIR = '../sols/rp_cache/'

# global parameters
hash_table = None
cache_dir = None


### CACHING FUNCTIONS

def generate_hash_table(direc=None, verbose=True):
    ''' '''

    global hash_table, cache_dir

    print('Generating hash table...')

    hash_table = {}

    if direc is None:
        cache_dir = CACHE_DIR
    else:
        cache_dir = direc

    cache_dir = os.path.normpath(cache_dir)+'/'

    print('\tDirectory: {0}'.format(cache_dir))

    # check that directory exists. If not, create.
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
        print('\tDirectory built... 0 hash values detected')
        return

    # get list of files in cache directory
    files = os.listdir(cache_dir)

    for f in files:
        key = os.path.splitext(f)[0]
        key = (-1 if key[0] == 'm' else 1)*int(key[1::])
        hash_table[key] = cache_dir + f

    print('\t{0} hash values detected'.format(len(hash_table)))


def to_cache(hash_pars, Es, prod_states, overwrite=False):
    '''Write a solution to the cache directory given its hash parameters'''

    global hash_table, cache_dir

    # rename parameters for convenience
    key = hash_pars['key']
    S = hash_pars['S']
    sh = hash_pars['sh']
    inds = hash_pars['inds']

    # check if target file already exists
    if not overwrite:
        if key in hash_table:
            print('Hash value already used...')

    # map solution to standard form using hash parameters
    _Es = [E/S for E in Es]
    _prod_states = []

    # construct write dict for export algorithm

    # write to file
    f = ('m' if key < 0 else '')+str(abs(key))

    # update hash table
    hash_table[key] = cache_dir + f

    pass


def from_cache(hash_pars):
    '''Load a solution from the cache directory given its hash parameters'''

    global hash_table, cache_dir

    key = hash_pars['key']

    if not key in hash_table:
        print('Hash value not found in hash table...')
        return

    # retrieve solutions dict from cache directory

    # map from standard form using hash parameters
    pass


### PARAMETER COMPUTATION FOR COMPONENT FORMALISM

def comp_on_comp(h, J, gam, modes):
    '''Compute the on-comp parameters for the given modes'''

    h = np.matrix(h)
    J = np.matrix(np.triu(J))

    N = h.shape[1]

    sz = h*modes.T
    sz2 = np.diag(modes*J*modes.T)

    if gam == 0.:
        g = None
    else:
        mds = np.asarray(modes)
        diff = np.abs(mds.reshape([-1, 1, N])-mds.reshape([1, -1, N]))
        g = gam*(np.sum(diff, axis=2) == 2)

    return [sz, sz2, g]


def comp_off_comp(J, mds1, mds2):
    '''Compute the comp-comp mode coupling parameters'''

    JJ = mds1*J*mds2.T

    return JJ


def partition(h, J, nparts):
    '''Split graph into nparts partitions: returns the indices, h and J
    parameters for each partition as well as coupling parameters between each
    partition'''

    # networkx graph
    G = nx.Graph(J)

    # connectivity list
    conn = G.adjacency_list()

    # partition in nparts
    ncuts, labels = part_graph(nparts, adjacency=conn)

    # indices of each partition
    parts = []
    for v in xrange(nparts):
        parts.append([i for i, x in enumerate(labels) if x == v])

    # make sure indices are sorted
    parts = map(sorted, parts)

    # make h, J for each partition
    h = np.array(h)
    h_p = [h[parts[i]] for i in xrange(nparts)]
    J_p = [J[parts[i], :][:, parts[i]] for i in xrange(nparts)]

    # coupling matrices
    C_p = {}
    for i in xrange(nparts):
        for j in xrange(i+1, nparts):
            C_p[(i, j)] = J[parts[i], :][:, parts[j]]

    return parts, h_p, J_p, C_p


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
            ps = tuple(np.array(map(int, bstr))*2-1)
            prod_states.append(ps)

    return prod_states


def proc_solve(e_vals, e_vecs, e_res):
    '''Process output from 'solve' function to match rp_solve'''

    states = e_vecs
    n_states = states.shape[1]

    states = [states[:, i] for i in xrange(n_states)]

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

    return Es, states, prod_states


def select_modes(Es, PS):
    '''Based on the energy gaps for the solution of each partition, select
    which product state modes are included. Input modes should be formated
    as vectors of +- 1.'''

#    print Es
#    print PS
    N = len(Es)     # number of partitions

    i_p = [1]*N     # number of included indices for each partition
    m_p = [0]*N     # number of included modes for each partition

    prod = 1.

    # offset energies by ground state energies
    for p in xrange(N):
        Es[p] -= Es[p][0]

    # force inclusion of ground state modes
    for p in xrange(N):
        m_p[p] += len(PS[p][0])
        prod *= m_p[p]

    # check fill condition
    if prod > MEM_THRESH:
        print 'Not enough memory capacity to facilitate ground state inclusion'
        return None

    # determine number of new product states for each partition
    K = [[] for i in xrange(N)]
    for i in xrange(N):
        ps = set(PS[i][0])
        for j in xrange(1, len(PS[i])):
            temp = len(ps)
            ps.update(PS[i][j])
            K[i].append((Es[i][j], i, len(ps)-temp))

    # order K values for prioritising mode inclusion
    order = sorted(reduce(lambda x, y: x+y, K))

    # main inclusion loop
    check = [False]*N   # set True if term tejected
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

    for p in xrange(N):
        mds = []
        for i in xrange(i_p[p]):
            mds += PS[p][i]
        mds = sorted(set(mds))
        modes.append(np.matrix(mds))

    if VERBOSE:
        print '%d modes selected...' % prod
        print 'mode dist: %s' % str(m_p)

    return modes


def gen_comp_diag(sz, sz2, J_m):
    '''Generate the diagonal terms in the component formalism Hamiltonian'''

    N = len(sz)     # number of components

    if VERBOSE:
        print 'Constructing diagonal component formalism parameters'
    # generate size counters
    M = [sz[p].shape[1] for p in xrange(N)]  # number of modes per partition
    C = np.cumprod(M)   # cumulative product of mode sizes
    C = np.insert(C, 0, 1)  # useful for avoiding special case

    # on comp energy terms
    if VERBOSE:
        print '\tComputing on-comp energy terms...'
    on_site = np.zeros([1, C[-1]], dtype=float)
    for p in xrange(N):
        a = sz[p] + sz2[p]
        temp = np.tile(np.repeat(a, C[p]), C[-1]/C[p+1])
        on_site += temp

    # external comp-comp mode coupling
    if VERBOSE:
        print '\tComputing comp-comp mode coupling terms...'
    ext_coup = np.zeros([1, C[-1]], dtype=float)
    for p1 in xrange(N):
        for p2 in xrange(p1+1, N):
            # handle p1 behaviour
            j = J_m[(p1, p2)].T
            if not j.any():
                continue
            #j = np.arange(j.size).reshape(j.shape)
            a = np.tile(np.repeat(j, C[p1], axis=1), C[p2]/C[p1+1])
            # handle p2 behaviour
            b = np.tile(a.flatten(), C[-1]/C[p2+1])
            ext_coup += b
    if VERBOSE:
        print '\t...done'
    return on_site + ext_coup


def gen_comp_tunn(G):
    '''Generate the off-diagonal tunneling terms in the component formalism
    Hamiltonian'''

    if VERBOSE:
        print 'Constructing off-diagonal component formalism parameters'

    N = len(G)  # number of partitions
    Nm = [x.shape[0] for x in G]    # number of modes per partition
    Cm = np.insert(np.cumprod(Nm), 0, 1)     # size of each partition sub-block

    # for each mode update the 'diagonal' submatrix
    mat = sp.coo_matrix((1, 1), dtype=float)
    for p in xrange(N):
        # construct the sub-block container
        A = []
        for m in xrange(Nm[p]):
            a = [None]*m + [mat]
            for n in xrange(m+1, Nm[p]):
                if G[p][n, m] == 0:
                    a.append(None)
                else:
                    dat = np.repeat(-G[p][n, m], Cm[p])
                    a.append(sp.diags([dat], [0]))
            A.append(a)
        mat = sp.bmat(A)

    if VERBOSE:
        print '\t...done'
    return mat.T


def general_decomp(n, c):
    '''Decompose a number into a general basis cumprod c'''

    rep = []
    for i in xrange(len(c)):
        t, n = divmod(n, c[i])
        rep.append(t)

    return rep


def correct_prod_state(pstates, modes, inds):
    '''Correct product state to account for mode space representation'''

    t = time()
    if VERBOSE:
        print 'Correcting product states...',

    # prep work

    Nps = len(pstates)  # number of product state lists
    N = len(modes)      # number of partitions

    inds = reduce(lambda x, y: x+y, inds)

    nmodes = [mds.shape[0] for mds in modes]    # number of modes per part
    C = np.cumprod([1]+nmodes[:-1])[::-1]     # cumprod of mode counts

    ps_modes = []
    # for each product state list
    for i in xrange(Nps):
        ps_list = pstates[i]
        ps_mds = []
        for ps in ps_list:
            rep = general_decomp(ps, C)[::-1]
            ps_m = [modes[j][rep[j]].tolist()[0] for j in xrange(N)]
            ps_m = reduce(lambda x, y: x+y, ps_m)
            # reorder using indices
            ps_mds.append(tuple([x[1] for x in sorted(zip(inds, ps_m))]))
        ps_modes.append(ps_mds)

    if VERBOSE:
        print 'done'
        print 'correct prod time: %.5f s' % (time()-t)
    return ps_modes


def proc_comp_solve(sols, modes, inds, e_res):
    '''Process the output of the sparse solver for the component formalism
    Hamiltonian'''

    t = time()
    if VERBOSE:
        print '\nProcessing component solver...'

    spec = sols['eigsh']['vals']

    states = sols['eigsh']['vecs']
    n_states = states.shape[1]

    if VERBOSE:
        print '\n\n%d states solved...' % n_states
    states = [states[:, i] for i in xrange(n_states)]

    Es = list(spec)
    prod_states = [get_prod_states(state, comp=True)
                   for state in states]

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

    # correct states to account for mode space
    prod_states = correct_prod_state(prod_states, modes, inds)

    if VERBOSE:
        print '\t...done'
        print 'Proc comp time: %.5f s' % (time()-t)

    return Es, states, prod_states


def proc_states(states, modes, inds):
    ''' '''

    # for each state, find significant indices, map to product states
    prod_states = [get_prod_states(state, comp=True) for state in states]
    amps = []
    for i in xrange(len(states)):
        amps.append(states[i][prod_states[i]])
    prod_states = correct_prod_state(prod_states, modes, inds)

    # express each state as a list of amplitudes and
    states = [[amps[i], prod_states[i]] for i in xrange(len(states))]

    return states


def build_comp_H(h_p, J_p, C_p, gam, modes):
    '''Build a sparse representation of the component space Hamiltonian'''

    N = len(h_p)    # number of partitions

    # get on component parameter
    outputs = [comp_on_comp(h_p[p], J_p[p], gam, modes[p]) for p in xrange(N)]
    sz, sz2, g = map(list, zip(*outputs))

    # get component mode coupling parameters
    J_m = {}
    for i in xrange(N):
        for j in xrange(i+1, N):
            J_m[(i, j)] = comp_off_comp(C_p[(i, j)], modes[i], modes[j])

    # construct diagonal elements
    diag = gen_comp_diag(sz, sz2, J_m)
    diag = sp.diags([diag[0]], [0])

    # construct tunneling elements as sparse matrix
    if not g[0] is None:
        off_diag = gen_comp_tunn(g)
        H = diag + off_diag
    else:
        H = diag

    return H


def solve_comp(h_p, J_p, C_p, gam, modes, inds, e_res):
    '''Solve component formalism'''

    if VERBOSE:
        print '\nRunning component solver...'
    t = time()

    H = build_comp_H(h_p, J_p, C_p, gam, modes)

    # run sparse matrix solver
    if VERBOSE:
        print 'H matrix size: %s' % str(H.shape)
        print 'Running sparse solver...',
    t1 = time()
    sols = solveSparse(H)
    if VERBOSE:
        print 'done'
        print 'sparse solver time: %.5f s' % (time()-t1)

    # process output
    Egaps, states, prod_states = proc_comp_solve(sols, modes, inds, e_res)

    #pprint(prod_states[0:2])
    #print Egaps
    if VERBOSE:
        print '...done'
        print 'Component solver time: %.5f s' % (time()-t)
    return Egaps, states, prod_states


def echo_ps(ps):
    '''Nice output format for product states'''
    s = ''.join(['+' if p < 0 else '-' for p in ps])
    return s


def rp_state_to_pol(amps, prod_states=None):
    '''convert the amplitudes and product states of a state to a pol list'''

    if prod_states is None:
        return state_to_pol(amps)

    modes = np.array(prod_states)
    amps = amps.reshape([-1, 1])

    return -np.sum((amps*amps)*modes, 0)


def out_handler(h, J, gam, prod_states):
    '''Make an estimation of low energy spectrum using the determined
    applicable subset of product states and the problem parameters'''

    # create a single list of the product states
    prod_states = [list(ps) for ps in prod_states]
    pstates = sorted(set(reduce(lambda x, y: x+y, prod_states)))

    # find the energy associated with each product state
    modes = np.matrix(pstates)
    H = build_comp_H([h], [J], [], gam, [modes])
    Eps = H.diagonal()

    # sort product states by energy
    ps_order = sorted(list(enumerate(Eps)), key=lambda x: x[1])
    ps_order, Eps = zip(*ps_order)

    # find the eigenstates in the reduced mode space
    e_vals, e_vecs = solve_sparse(H)

    E = e_vals
    states = e_vecs
    states = [states[ps_order, i] for i in xrange(len(E))]

    # get cell polarizations for each eigenstate
    state_pols = [rp_state_to_pol(state, pstates) for state in states]

    return E, states, pstates


def rp_solve(h, J, gam, rec=False, cache_direc=None):
    '''Solve transverse field ising spin-glass configuration using recursive
    partitioning with low-energy spectrum mode composition'''

    global cache_dir, hash_table

    if VERBOSE:
        print 'Detected problem size: %d...' % len(h)

    if CACHE:
        # initialise hash table
        if hash_table is None:
            generate_hash_table(direc=cache_direc)

        # generate hash parameters
        hash_pars = hash_problem(h, J, gam)

        # check for previous solution
        if hash_pars['key'] in hash_table:
            Es, prod_states = from_cache(hash_pars)
            if not rec:
                return out_handler(h, J, gam, prod_states)
            return Es, prod_states

    ## cache check cleared, must solve recursion step

    e_res = np.max(np.abs(J))*E_RES

    # format h
    h = np.asarray(np.reshape(h, [1, -1])).tolist()[0]

    if len(h) <= N_THRESH:
        if VERBOSE:
            print 'Running exact solver...'
        e_vals, e_vecs = solve(h, J, gamma=gam, more=True)
        Es, states, prod_states = proc_solve(e_vals, e_vecs, e_res)
        states = zip(states, [None]*len(states))
    else:
        if VERBOSE:
            print 'Running recursive partition...'
        # ensure J not triu
        J = np.triu(J)+np.triu(J, 1).T

        # partition into some number of subgraphs
        nparts = int(min(ceil(len(h)*1./N_THRESH), N_PARTS))
        parts, h_p, J_p, C_p = partition(h, J, nparts)

        # solve each partition
        ES = []
        PS = []
        for i in xrange(nparts):
            Es, prod_states = rp_solve(h_p[i], J_p[i], gam, rec=True)
            ES.append(Es)
            PS.append(prod_states)

        # select modes to include
        modes = select_modes(ES, PS)

        # solve component system
        Es, states, prod_states = solve_comp(h_p, J_p, C_p, gam,
                                             modes, parts, e_res)

        # if initial call (last after recursion), format state outputs
#        if not rec:
#            states = proc_states(states, modes, parts)

    # write to cache
    if CACHE:
        to_cache(hash_pars, Es, prod_states, overwrite=OVERWRITE)

    if not rec:
        return out_handler(h, J, gam, prod_states)

    return Es, prod_states


if __name__ == '__main__':
    pass
