
from __future__ import print_function

import numpy as np
import networkx as nx
import scipy.sparse as sp

import os, json, re

from collections import Iterable
from numbers import Number

from bcp import chlebikova
from itertools import combinations

from sparse import solve_sparse
import core

from pprint import pprint

# solver parameters
N_THRESH = 6            # largest allowed number of nodes for exact solver
MEM_THRESH = 1e5        # maximum mode count produce
STATE_THRESH = 0.05     # required aplitude for state contribution

E_RES = 1e-3            # resolution in energy binning
W_POW = 1.              # power for weighing nodes in chlebikova bisection

CACHING = False

# general functions

def compute_rp_tree(J, inds=None):
    '''Build the recursive partition tree of the adjacency matrix J using the
    Chlebikova heuristic BCP2 method'''

    if inds is None:
        inds = range(J.shape[0])

    tree = {'inds': inds, 'children': []}
    if J.shape[0] <= N_THRESH:
        return tree

    G = nx.Graph(J!=0)
    for k in G:
        G.node[k]['w'] = 1./len(G[k])**W_POW

    try:
        V1, V2 = chlebikova(G)
    except AssertionError:
        print('Chlebikova failed. The problem is likely disjoint.')

    parts = [sorted(x) for x in [V1, V2]]   # incidices in each partition

    for part in parts:
        sub_tree = compute_rp_tree(J[part,:][:,part], inds=part)
        tree['children'].append(sub_tree)

    return tree


def generate_hash_table(direc=None):
    '''If direc exists, generate the has table from the directory fil structure.
    Otherwise, create the given directory'''

    if direc is None:
        return None

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


class RP_Node:
    ''' '''

    def __init__(self, h, J, gam, tree, nx, nz, **kwargs):
        '''Initialise a RP Solver Node.'''

        # define parameters (for C porting)

        self.h = h              # local h parameters
        self.J = J              # local J parameters
        self.gam = gam          # local gamma parameters

        self.nx = nx            # basis x-fields
        self.nz = nz            # basis z-fields

        self.vprint = None      # print method

        self.tree = tree        # recursion tree at current node

        self.cache = None       # cache dictionary
        self.hash_pars = None   # hashing parameters

        self.Hx = None          # local off-diagonal Hamiltonian
        self.Hz = None          # local diagonal Hamiltonian

        self.Es = None          # energy bins
        self.modes = None       # new z-modes in each energy bin

        self.children = None    # pointers to children nodes
        self.pmodes = None      # modes included from each child

        if 'vprint' in kwargs:
            self.vprint = kwargs['vprint']
        else:
            self.vprint = lambda *a, **k: None

        # detect optional parameters
        if 'cache' in kwargs:
            self.cache = kwargs['cache']

        # construct children nodes
        self.vprint('Constructing children...')
        self.children = []
        for child in tree['children']:
            inds = child['inds']
            _h = self.h[inds]
            _J = self.J[inds,:][:,inds]
            if self.gam is None:
                _gam = None
            else:
                _gam = self.gam[inds]
            _nx = self.nx[inds]
            _nz = self.nz[inds]
            node = RP_Node(_h, _J, _gam, child, _nx, _nz,
                            cache=self.cache, vprint=self.vprint)
            self.children.append(node)

    # caching methods
    def from_cache(self):
        ''' '''
        pass

    def to_cache(self):
        '''Convert solved parameters to standard form and save to cache'''

        # map hashval to a filename for storage
        hval = self.hash_pars['hval']
        ext = '{0}{1}.json'.format('m' if hval<0 else 'p', abs(hval))
        fname = os.path.join(self.cache['dir'], ext)

    # eigendecomposition processing methods

    def get_prod_states(self, state, comp=False):
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

    def proc_exact_solve(self, e_vals, e_vecs, e_res):
        '''Process output from exact solve. Bin energies and record new modes
        for each bin'''

        n_states = e_vecs.shape[1]

        states = [e_vecs[:, i] for i in range(n_states)]

        # associate energies with product states

        Es = list(e_vals)
        prod_states = [self.get_prod_states(state) for state in states]

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

    def select_modes(self):
        '''Based on the energy gaps for the solution of each partition, select
        which product state modes are included.'''

        Np = len(self.children)    # number of partitions

        i_p = [1]*Np     # number of included indices for each partition
        m_p = [0]*Np     # number of included modes for each partition

        prod = 1        # number of modes in product space

        # offset energies by ground state energies
        ES = [child.Es-child.Es[0] for child in self.children]
        MS = [child.modes for child in self.children]

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

    # Hamiltonian formulations

    def exact_formulation(self):
        '''Construct an exact Hamiltonian for the problem node. Return the
        off-diagonal elements Hx and diagonal elements Hz. If self.gam is None
        then Hx will be None. Otherwise it will be a sparse matrix. Hz will be
        a 1D array.'''

        if self.gam is None:
            Hx = None
            Hz = generate_Hz(self.h, self.J)
            return Hx, Hz

        # compute prefactors for simple terms
        gam = self.gam*self.nz + self.h*self.nx
        h = -self.gam*self.nx + self.h*self.nz

        # compute prefactors for complex terms
        chi = self.J*np.outer(self.nx, self.nx)
        J = self.J*np.outer(self.nz, self.nz)
        Gam = self.J*np.outer(self.nx, self.nz)

        # standard Ising components
        Hx = core.generate_Hx(gam)
        Hz = core.generate_Hz(h, J)

        # compute additional diagonal elements
        Hxx = core.generate_Hxx(chi)
        Hxz = core.generate_Hxz(Gam)

        Hx = Hx + Hxx - Hxz

        return Hx, Hz

    def solve(self):
        '''Solve the problem node, recursively solves all children'''

        self.vprint('Running Solver...')
        # check cache for solution
        if False and self.cache:
            # compute hash_parameters
            hval, K, hp, inds = core.hash_problem(h, J, gam=gam)
            self.hash_pars = {'hval': hval, 'K': K, 'hp': hp, 'inds': inds}
            # look in hash table
            if hval in self.cache['table']:
                try:
                    Es, modes = from_cache(self.cache['dir'], **self.hash_pars)
                except:
                    pass

        # solution not cached, compute
        e_res = np.max(np.abs(self.J))*E_RES    # proportional energy resolution

        # if small enough, solve exactly. Otherwise solve recursively.
        if not self.tree['children']:
            self.vprint('Running exact solver...')
            # construct matrix elements
            Hx, Hz = self.exact_formulation()
            Hs = Hx + sp.diags(Hz)
            # store local Hamiltonian components
            self.Hx = Hx
            self.Hz = Hz
            # solve
            e_vals, e_vecs = solve_sparse(Hs, more=True)
            Es, modes = self.proc_exact_solve(e_vals, e_vecs, e_res)

        else:
            self.vprint('Running recursive solver')
            # solve each child
            for child in self.children:
                child.solve()
            # select modes to keep from each child
            modes = self.select_modes()
            # reduce the Hilbert space of the children Hamiltonians
            for mds, child in zip(modes, self.children):
                pprint(mds)
                print(child.Hx.shape)
                print(child.Hz.shape)
                print(len(mds))

            # formulate local Hamiltonian
            # solve

        self.Es = Es
        self.modes = modes




class RP_Solver:
    ''' '''

    def __init__(self, h, J, gam, **kwargs):
        '''Initialise an RP-Solver instance.'''

        # define parameters (for C porting)
        self.h = None
        self.J = None
        self.gam = None

        self.nx = None
        self.nz = None

        self.cache = {}

        # set verbose print method
        if 'verbose' in kwargs and kwargs['verbose']:
            self.vprint = print
        else:
            self.vprint = lambda *a, **k: None

        # store caching elements

        if 'cache_dir' in kwargs:
            self.cache['dir'] = kwargs['cache_dir']
        else:
            self.cache['dir'] = None

        if 'hash_table' in kwargs:
            self.cache['table'] = kwargs['hash_table']
        else:
            self.cache['table'] = generate_hash_table(self.cache['dir'])

        # regularize input format
        self.vprint('Standardizing input format...')

        assert isinstance(h, Iterable), 'h is not iterable'
        h = np.array(h).reshape([-1,])

        assert isinstance(J, Iterable), 'J is not iterable'
        J = np.array(J)
        assert J.shape == (len(h), len(h)), 'J is the wrong shape'

        if isinstance(gam, Number):
            gam = np.ones([len(h),])*float(gam)
        elif isinstance(gam, Iterable):
            gam = np.array(gam).reshape([-1,])
            assert len(gam) == len(h), 'gam is the wrong size'
        elif gam is not None:
            raise AssertionError('Invalid gamma format. Must be scalar, iterable, or None')

        if gam is not None and np.max(np.abs(gam)) < 1e-4*np.max(np.abs(J)):
            gam = None

        # force J to be symmetric
        if np.any(np.tril(J, -1)):
            J = J.T
        J = np.triu(J)+np.triu(J, 1).T

        # initialize recursion tree
        try:
            tree = kwargs['tree']
            assert check_tree(tree, ar=range(len(h))), '\t given tree is inavlid...'
        except (KeyError, AssertionError):
            self.vprint('\tconstructing recursion tree...')
            tree = compute_rp_tree(J != 0)

        if 'nx' in kwargs and 'nz' in kwargs:
            self.nx = kwargs['nx']
            self.nz = kwargs['nz']
        else:
            self.nx = np.zeros([len(h),])
            self.nz = np.ones([len(h),])

        # normalise nx and nz
        f = np.sqrt(self.nx**2 + self.nz**2)
        self.nx /= f
        self.nz /= f

        self.node = RP_Node(h, J, gam, tree, self.nx, self.nz,
                                cache=self.cache, vprint=self.vprint)

    def solve(self, **params):
        '''Run the RP_Solver on the specified problem. Functionality for
        additional solver parameters to be added.'''

        self.node.solve()
