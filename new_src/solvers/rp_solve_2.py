
from __future__ import print_function

import numpy as np
import networkx as nx
import scipy.sparse as sp

import os, json, re

from collections import Iterable, defaultdict
from numbers import Number

from bcp import chlebikova
from itertools import combinations

from sparse import solve_sparse
import core

from time import time
from pprint import pprint

# solver parameters
N_THRESH = 8            # largest allowed number of nodes for exact solver
MEM_THRESH = 1e5        # maximum mode count produce
STATE_THRESH = 0.02     # required aplitude for state contribution

E_RES = 1e-3            # resolution in energy binning
W_POW = 1.              # power for weighing nodes in chlebikova bisection

CACHING = True

# general functions

def tick(s, t):
    print('{0}: {1:.3f} (s)'.format(s, time()-t))
    return time()

def show_symmetry(M, s):
    ''' '''

    D = M-M.transpose()

    print('\n\n{0}\n'.format(s))
    for i,j in np.transpose(np.nonzero(D)):
        print('{0}:{1} :: {2}'.format(i, j, D[i,j]))
    print('\n\n')

def general_decomp(n, c):
    '''Decompose a number into a general basis cumprod c'''

    rep = []
    for i in range(len(c)):
        t, n = divmod(n, c[i])
        rep.append(t)

    return rep

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

# some matrix generations

def compute_px(i, modes):
    '''compute pauli_x(i) using a given matrix of modes'''
    # indices of all modes that differ only at spin i
    mds = np.asarray(modes)
    N = mds.shape[1]
    if N==1:
        px = np.array([[0,1],[1,0]])
    else:
        diff = np.abs(mds.reshape([-1, 1, N])-mds.reshape([1, -1, N]))
        px = (diff[:,:,i]*(np.sum(diff, axis=2)==2))/2
    return sp.coo_matrix(px)

def compute_pz(i, modes):
    '''compute pauli_z(i) using a given matrix of modes'''
    return np.array(modes[:,i]).reshape([-1,])



# field estimation methods

def soft_fields(h, J, gam):
    '''Compute soft bounds for nz. Assumes nz>0'''
    N = len(h)

    nx = np.zeros([N,])
    nz = np.ones([N,])

    F = 10
    if gam is not None:
        m = max(np.max(np.abs(h)), np.max(np.abs(J)))
        g = np.max(np.abs(gam))
        if m > F*g:
            pass
        elif g > F*m:
            nx, nz = nz, nx
        else:
            nx = -gam
            nz = -(np.abs(h) + .5*np.sum(np.abs(J), axis=1))

    return nx, nz



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

        self.px = {}            # precomputed pauli_x operators in mode space
        self.pz = {}            # precomputed pauli_z operators in mode space

        self.Es = None          # energy bins
        self.minds = None       # new mode indices in each energy bin
        self.modes = None       # currently included modes, lexic. sorted

        self.e_vals = None      # EVD eigenvalues
        self.e_vecs = None      # EVD eigenvectors

        self.children = None    # pointers to children nodes

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
        # simplify parameter name
        hval = self.hash_pars['hval']
        K = self.hash_pars['K']
        hp = self.hash_pars['hp']
        inds = self.hash_pars['inds']

        inv_map = {k: i for i, k in enumerate(inds)}
        ninds = [inv_map[k] for k in range(len(inds))]

        # map hashval to a filename for reading
        ext = '{0}{1}.json'.format('m' if hval<0 else 'p', abs(hval))
        fname = os.path.join(self.cache['dir'], ext)

        fp = open(fname, 'r')
        data = json.load(fp)
        fp.close()

        Es = np.array(data['Es'])
        minds = data['minds']
        modes = np.matrix(data['modes'])[:,ninds]
        nx = np.array(data['nx'])[ninds]
        nz = np.array(data['nz'])[ninds]

        Es = [E*K for E in Es]

        self.Es = Es
        self.minds = minds
        self.modes = modes
        self.nx = nx
        self.nz = nz

    def to_cache(self):
        '''Convert solved parameters to standard form and save to cache'''

        # simplify parameter names
        hval = self.hash_pars['hval']
        K = self.hash_pars['K']
        hp = self.hash_pars['hp']
        inds = self.hash_pars['inds']

        # map hashval to a filename for storage
        ext = '{0}{1}.json'.format('m' if hval<0 else 'p', abs(hval))
        fname = os.path.join(self.cache['dir'], ext)

        Es_ = [E/K for E in self.Es]

        modes = (hp*self.modes[:,inds]).tolist()

        nx_ = tuple(self.nx[inds])
        nz_ = tuple(self.nz[inds])

        data = {'Es': Es_,
                'minds': self.minds,
                'modes': modes,
                'nx': nx_,
                'nz': nz_}

        fp = open(fname, 'w')
        json.dump(data, fp)
        fp.close()

        return ext

    # eigendecomposition processing methods

    def get_prod_states(self, state):
        '''Get state indices of the product states which contribute to the
        given state.'''

        # number of spins
        N = int(np.log2(state.shape[0]))

        # isolate contributing product states
        inds = (np.abs(state) > STATE_THRESH).nonzero()[0]

        # sort by contribution magnitude
        inds = sorted(inds, key=lambda x: np.abs(state)[x], reverse=True)

        return inds

    def proc_solve(self, e_vals, e_vecs, e_res, comp=False, direct=False):
        '''Process the output of the sparse solver. Use comp=True is the
        component formalism was used.'''

        n_states = e_vecs.shape[1]

        states = [e_vecs[:, i] for i in range(n_states)]
        N = int(np.log2(states[0].shape[0]))

        if comp:
            nmodes = [child.modes.shape[0] for child in self.children]  # number of modes per child
            C = np.cumprod([1]+nmodes[-1:0:-1])[::-1]                       # cumprod of mode counts

        # associate energies with product states
        energies = list(e_vals)
        sinds = [self.get_prod_states(state) for state in states]

        # create index mapping
        all_inds = set()
        for inds in sinds:
            all_inds.update(inds)

        ind_map = {ind: n for n, ind in enumerate(sorted(all_inds))}

        # energy bins and new mode indices per bin
        Es = []
        minds = []

        all_inds = set()
        while energies:
            E = energies[0]
            try:
                i = next(i for i, x in enumerate(energies) if x > E+e_res)
            except:
                i = None

            inds = set(reduce(lambda x, y: x+y, sinds[:i]))
            new_inds = inds-all_inds
            all_inds.update(new_inds)

            Es.append(E)
            minds.append(sorted(new_inds))
            if i is None:
                energies = []
                sinds = []
            else:
                energies = energies[i:]
                sinds = sinds[i:]

        # remap minds
        for inds in minds:
            for i, x in enumerate(inds):
                inds[i] = ind_map[x]

        # keep track of the actual state indices
        inds = sorted(all_inds)

        # construct modes using true state indices
        modes = []
        for ind in inds:
            if direct:
                mode = self.modes[ind]
            elif comp:
                rep = general_decomp(ind, C)
                mode = []
                for x, child in enumerate(self.children):
                    mode += child.modes[rep[x]].tolist()[0]
            else:
                bstr = format(ind, '#0{0}b'.format(N+2))[2::]
                mode = tuple(1-2*np.array(list(map(int, bstr))))
            modes.append(mode)

        return Es, minds, modes, inds

    def select_modes(self):
        '''Based on the energy gaps for the solution of each partition, select
        which product state modes are included.'''

        Np = len(self.children)    # number of partitions

        i_p = [1]*Np     # number of included indices for each partition
        m_p = [0]*Np     # number of included modes for each partition

        prod = 1.        # number of modes in product space

        # offset energies by ground state energies
        ES = [child.Es-child.Es[0] for child in self.children]

        # store mode indices from each child
        MIS = [child.minds for child in self.children]

        # force inclusion of ground state modes
        for p, mis in enumerate(MIS):
            m_p[p] += len(mis[0])
            prod *= m_p[p]

        # check fill condition
        assert prod < MEM_THRESH, 'Not enough memory for ground state inclusion'

        # determine number of new product states for each partition
        K = []
        for p, (es, mis) in enumerate(zip(ES, MIS)):
            for e, inds in zip(es[1:], mis[1:]):
                K.append((e, p, len(inds)))

        order = sorted(K)

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

        # format the modes for output: list of mode matrice
        cinds = []

        for p, mis in enumerate(MIS):
            inds = []
            for i in range(i_p[p]):
                inds += mis[i]
            cinds.append(sorted(inds))

        return cinds

    # Hamiltonian formulations

    def compute_pz(self, i):
        '''compute the pauli z operator for the i^th spin of the local problem
        in the current mode space. Returns the diagonal. Spin labels start at
        0'''

        if i in self.pz:
            pz = self.pz[i]
        else:
            pz = compute_pz(i, self.modes)
            self.pz[i] = pz
        return pz

    def compute_px(self, i):
        '''compute the pauli x operator for the i^th spin of the local problem
        in the current mode space. Returns as sparse coo_matrix. Spin labels
        start at 0'''

        if i in self.px:
            px = self.px[i]
        else:
            px = compute_px(i, self.modes)
            self.px[i] = px

        return px

    def exact_formulation(self):
        '''Construct an exact Hamiltonian for the problem node. Return the
        off-diagonal elements Hx and diagonal elements Hz. If self.gam is None
        then Hx will be None. Otherwise it will be a sparse matrix. Hz will be
        a 1D array.'''

        if self.gam is None:
            Hx = None
            Hz = core.generate_Hz(self.h, self.J)
            return Hx, Hz

        # compute prefactors for simple terms
        gam = self.gam*self.nz + self.h*self.nx
        h = -self.gam*self.nx + self.h*self.nz

        # compute prefactors for complex terms
        chi = self.J*np.outer(self.nx, self.nx)
        J = self.J*np.outer(self.nz, self.nz)
        Gam = self.J*np.outer(self.nx, self.nz)

        # standard Ising components
        Hx = -core.generate_Hx(gam[::-1])
        Hz = core.generate_Hz(h, J)

        # compute additional diagonal elements
        Hxx = core.generate_Hxx(chi)
        Hxz = core.generate_Hxz(Gam)

        Hx = Hx + Hxx - Hxz

        return Hx, Hz

    def comp_formulation(self):
        '''Construct the local Hamiltonian using the component formulation. If
        self.gam is None then Hx will be None. Otherwise it will be a sparse
        matrix. Hz will be diagonal'''

        f = lambda x: np.round(x.todense(), 2)

        # compute child-child subsets of J
        CJ, CChi, CGam = {}, {}, {}
        for p1, p2 in combinations(range(len(self.children)),2):
            c1 = self.tree['children'][p1]
            c2 = self.tree['children'][p2]
            i1, i2 = c1['inds'], c2['inds']
            Jl = self.J[i1,:][:,i2]
            Jr = self.J[i2,:][:,i1]
            CJ[(p1,p2)] = Jl*np.outer(self.nz[i1], self.nz[i2])
            CChi[(p1,p2)] = Jl*np.outer(self.nx[i1], self.nx[i2])
            CGam[(p1,p2)] = Jl*np.outer(self.nx[i1], self.nz[i2])
            CGam[(p2,p1)] = Jr*np.outer(self.nx[i2], self.nz[i1])

        ### compute diagonal components

        # from local Hz of children
        Hz = core.multi_kron_sum(*[child.Hz for child in self.children])

        # zz partition interaction terms
        Hzz = np.zeros(Hz.shape)

        # only works for 2 partitions
        for (p1, p2), C in CJ.iteritems():
            for i1, i2 in np.transpose(np.nonzero(C)):
                pz1 = self.children[p1].compute_pz(i1)
                pz2 = self.children[p2].compute_pz(i2)
                Hzz += C[i1,i2]*np.kron(pz1, pz2)

        Hz += Hzz


        if self.gam is None:
            Hx = None
        else:
            cHx = []
            for child in self.children:
                if child.Hx is None:
                    N = child.Hz.size
                    cHx.append(sp.coo_matrix((N,N)))
                else:
                    cHx.append(child.Hx)
            Hx = core.multi_kron_sum(*cHx)

            # xx partition interaction terms
            Hxx = sp.coo_matrix(Hx.shape)

            # only works for 2 partitions
            for (p1, p2), C in CChi.iteritems():
                for i1, i2 in np.transpose(np.nonzero(C)):
                    px1 = self.children[p1].compute_px(i1)
                    px2 = self.children[p2].compute_px(i2)
                    Hxx += C[i1,i2]*sp.kron(px1, px2)

            # xz partition interactions terms
            Hxz = sp.coo_matrix(Hx.shape)

            # only works for 2 partitions
            for (p1, p2), C in CGam.iteritems():
                for i1, i2 in np.transpose(np.nonzero(C)):
                    px1 = self.children[p1].compute_px(i1)
                    pz2 = self.children[p2].compute_pz(i2)
                    if p1<p2:
                        Hxz += C[i1,i2]*sp.kron(px1, sp.diags(pz2))
                    else:
                        Hxz += C[i1,i2]*sp.kron(sp.diags(pz2), px1)

            Hx += Hxx - Hxz

        return Hx, Hz

    def direct_construction(self):
        '''Construct the local parameters directly from the problem parameters
        without recursion. Requires the modes to be known'''

        N, Nm = len(self.h), self.modes.shape[0]

        # compute all the pauli operators
        PX = [self.compute_px(i) for i in range(N)]
        PZ = [self.compute_pz(i) for i in range(N)]

        f = lambda x: np.round(x, 2)

        # compute effective local fields
        if self.gam is None:
            h = self.h*self.nz
            gam = self.h*self.nx
        else:
            h = -self.gam*self.nx + self.h*self.nz
            gam = self.gam*self.nz + self.h*self.nx

        # compute effective interaction terms
        J = self.J*np.outer(self.nz, self.nz)
        Chi = self.J*np.outer(self.nx, self.nx)
        Gam = self.J*np.outer(self.nx, self.nz)

        # z terms
        Hz = np.zeros([Nm,])
        for i in h.nonzero()[0]:
            Hz += h[i]*PZ[i]

        # zz terms
        Hzz = np.zeros([Nm,])
        for i, j in np.transpose(J.nonzero()):
            if i>j:
                continue
            Hzz += J[i,j]*PZ[i]*PZ[j]

        Hz += Hzz

        if self.gam is None:
            Hx = None
        else:
            # x terms
            Hx = sp.coo_matrix((Nm,Nm))
            for i in gam.nonzero()[0]:
                Hx += gam[i]*PX[i]

            # xx terms
            Hxx = sp.coo_matrix((Nm,Nm))
            for i,j in np.transpose(Chi.nonzero()):
                if i>j:
                    continue
                Hxx += Chi[i,j]*PX[i]*PX[j]

            # xz terms
            Hxz = sp.coo_matrix((Nm,Nm))
            for i,j in np.transpose(Gam.nonzero()):
                Hxz += Gam[i,j]*PX[i]*sp.diags(PZ[j])

            Hx = sp.tril(-Hx + Hxx - Hxz)
            Hx = Hx + Hx.T

        return Hx, Hz

    def evd(self):
        ''' '''

        if self.e_vals is None or self.e_vecs is None:
            Hs = self.Hx+sp.diags(self.Hz)
            e_vals, e_vecs = solve_sparse(Hs, more=False)
            self.e_vals = e_vals
            self.e_vecs = e_vecs

        return self.e_vals, self.e_vecs

    # field estimation metho
    # solvers

    def mode_solve(self, modes):
        '''Solve the node in the basis of known modes'''

        e_res = np.max(np.abs(self.J))*E_RES

        f = lambda x: np.round(x, 5)

        _Hz = np.array(self.Hz)
        _Hx = sp.coo_matrix(self.Hx)

        self.modes = modes
        Hx, Hz = self.direct_construction()
        self.Hx = Hx
        self.Hz = Hz

        if Hx is None:
            Hs = sp.diags(Hz)
        else:
            Hs = Hx + sp.diags(Hz)

        e_vals, e_vecs = solve_sparse(Hs, more=False)
        Es, minds, modes, inds = self.proc_solve(e_vals, e_vecs, e_res)

        # reduce Hamiltonian

        if self.Hx is not None:
            self.Hx = self.Hx[inds,:][:,inds]
        self.Hz = self.Hz[inds]

        # store data

        self.Es = np.array(Es)
        self.minds = minds
        self.modes = np.matrix(modes)

        self.e_vals = e_vals
        self.e_vecs = e_vecs[inds,:]

    def solve(self):
        '''Solve the problem node, recursively solves all children'''

        self.vprint('Problem size: {0}...'.format(len(self.h)))

        # check cache for solution
        if CACHING and self.cache['dir']:
            # compute hash_parameters
            hval, K, hp, inds = core.hash_problem(self.h, self.J, gam=self.gam,
                                                    nx=self.nx, nz=self.nz)
            self.hash_pars = {'hval': hval, 'K': K, 'hp': hp, 'inds': inds}
            # look in hash table
            if hval in self.cache['table']:
                try:
                    self.from_cache()
                    Hx, Hz = self.direct_construction()
                    self.Hx = Hx
                    self.Hz = Hz
                    return
                except Exception as e:
                    print(e.message)
                    print('Something went wrong load from cache')

        # solution not cached, compute
        e_res = np.max(np.abs(self.J))*E_RES    # proportional energy resolution

        # if small enough, solve exactly. Otherwise solve recursively.
        if not self.tree['children']:
            self.vprint('Running exact solver...')
            t = time()
            # construct matrix elements
            Hx, Hz = self.exact_formulation()
            if Hx is None:
                Hs = sp.diags(Hz)
            else:
                Hs = Hx + sp.diags(Hz)
            # store local Hamiltonian components
            self.Hx = Hx
            self.Hz = Hz
            # solve
            e_vals, e_vecs = solve_sparse(Hs, more=True)
            Es, minds, modes, inds = self.proc_solve(e_vals, e_vecs, e_res)

        else:
            self.vprint('Running recursive solver')

            # solve each child
            for child in self.children:
                child.solve()

            # select mode indices to keep from each child
            cinds = self.select_modes()

            # reduce the Hilbert space of the children Hamiltonians
            for inds, child in zip(cinds, self.children):
                # reduce operators
                if child.Hx is not None:
                    child.Hx = child.Hx[inds,:][:,inds]
                child.Hz = child.Hz[inds]
                # reduce number of modes
                child.modes = child.modes[inds]
                # forget old pauli matrices
                child.px = {}
                child.pz = {}

            # formulate local Hamiltonian
            Hx, Hz = self.comp_formulation()

            if Hx is None:
                Hs = sp.diags(Hz)
            else:
                Hs = Hx + sp.diags(Hz)

            # store local Hamiltonian components
            self.Hx = Hx
            self.Hz = Hz
            # solve

            e_vals, e_vecs = solve_sparse(Hs, more=False)
            Es, minds, modes, inds = self.proc_solve(e_vals, e_vecs, e_res, comp=True)

        # reduce local Hamiltonians
        if self.Hx is not None:
            self.Hx = self.Hx[inds,:][:, inds]
        self.Hz = self.Hz[inds]

        # formatting and storage
        self.Es = np.array(Es)
        self.minds = minds
        self.modes = np.matrix(modes)

        self.e_vals = e_vals
        self.e_vecs = e_vecs[inds,:]

        # delete references to children to free memory
        self.children = None

        if CACHING and self.cache['dir'] and self.hash_pars['hval'] not in self.cache['table']:
            try:
                self.cache['table'][self.hash_pars['hval']] = self.to_cache()
            except Exception as e:
                print(e.message)
                print('Failed to cache solution...')

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
            nx, nz = soft_fields(h, J, gam)
            self.nx = nx
            self.nz = nz

        # normalise nx and nz
        f = np.sqrt(self.nx**2 + self.nz**2)
        self.nx /= f
        self.nz /= f

        # print('nx: {0}'.format(np.round(self.nx,2)))
        # print('nz: {0}'.format(np.round(self.nz,2)))

        self.node = RP_Node(h, J, gam, tree, self.nx, self.nz,
                                cache=self.cache, vprint=self.vprint)

    # field estimation methods

    def get_current_fields(self):
        '''Access the nx, nz fields for the problem node'''

        return self.nx, self.nz

    def ground_fields(self):
        '''Compute nx, nz using <pz> from the estimated ground state'''

        node = self.node
        N = len(node.h)

        if node.gam is None:
            return np.zeros([N,]), np.ones([N,])

        e_vals, e_vecs = node.evd()
        ground = e_vecs[:,0]
        amps = np.abs(ground)**2

        PZ = np.sum(np.asarray(node.modes)*amps.reshape([-1,1]), axis=0)
        PX = []

        for i in range(N):
            px = node.compute_px(i)
            s = 0.
            for n,m in np.transpose(px.nonzero()):
                s += ground[n]*np.conj(ground[m])
            PX.append(s)

        PX = np.array(PX).reshape([-1,])
        PZ = np.array(PZ).reshape([-1,])
        EXP = -node.nx*PX + node.nx*PZ

        nx = -node.gam
        nz = node.h+.5*np.dot(node.J,EXP)

        return nx, nz

    def mode_solve(self, modes):
        '''Solve the problem directly in the basis of given modes'''

        modes_ = np.matrix(modes)
        self.node.mode_solve(modes_)

    def solve(self, **params):
        '''Run the RP_Solver on the specified problem. Functionality for
        additional solver parameters to be added.'''

        self.node.solve()
