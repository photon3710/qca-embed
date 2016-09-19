
import numpy as np
import scipy.sparse as sp
import networkx as nx

import os, json, re

from collections import Iterable
from numbers import Number
from pymetis import part_graph
from bcp import chlebikova
from itertools import combinations
from time import time

from core import hash_problem, state_to_pol
from sparse import solve as exact_solve
from sparse import solve_sparse

# solver parameters
N_PARTS = 2     # number of partitions at each recursive step

N_THRESH = 4            # largest allowed number of nodes for exact solver
MEM_THRESH = 1e5        # maximum mode count product
STATE_THRESH = 0.05     # required amplitude for state contribution
E_RES = 1e-3            # resolution in energy binning
W_POW = 1               # power for weighting nodes in chlebikova bisection

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

# default mixer method

def default_mixer(gam, eps):
    '''Default method for computing the mixing coefficients'''
    if eps == 0 or gam/eps > 1:
        return (1, 0)
    else:
        return (0, 1)

### CLASSES

class RP_Solver:
    '''Solver class for the RP method. '''

    def __init__(self, h, g, J, **kwargs):
        ''' '''

        # set verbose print method
        if 'verbose' in kwargs and kwargs['verbose']:
            self.vprint = print
        else:
            self.vprint = lambda *a, **k: None

        # store cache directory
        if 'cache_dir' in kwargs:
            self.cache_dir = kwargs['cache_dir']
        else:
            self.cache_dir = None

        # assign hash table
        if 'hash_table' in kwargs:
            self.hash_table = kwargs['hash_table']
        else:
            try:
                self.generate_hash_table(self.cache_dir)
            except:
                self.hash_table = {}

        # regularize input formatting
        self.vprint('Standardizing input format...')

        assert isinstance(h, Iterable), 'h is not iterable'
        h = np.array(h).reshape([-1,])

        assert isinstance(J, Iterable), 'J is not iterable'
        J = np.array(J)
        assert J.shape == (len(h), len(h)), 'J is the wrong shape'

        if isinstance(gam, Number):
            gam = float(gam)
        elif gam is not None:
            raise AssertionError('Invalid gamma format. Must be scalar or None')

        # force J to by symmetric
        if np.any(np.tril(J, -1)):
            J = J.T
        J = np.triu(J)+np.triu(J,1)

        # initialise recursion tree
        self.vprint('Getting recursion tree')
        try:
            tree = kwargs['tree']
            assert check_tree(tree, ar=range(h)), '\tgiven tree invalid'
        except (KeyError, AssertionError):
            self.vprint('\trunning graph clustering...')
            tree = compute_rp_tree(J != 0, nparts = N_PARTS)

        self.problem = RP_Problem(self, h, gam, J, tree)

    def generate_hash_table(self, direc):
        ''' '''

        hash_table = {}

        if direc is not None:
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

        self.hash_table = hash_table

    def solve(self):
        ''' '''
        pass


class RP_Problem:
    ''' '''

    def __init__(self, solver, h, g, J, tree, mix, **kwargs):
        ''' '''

        self.solver = solver
        self.vprint = solver.vprint

        self.h = h          # local fields
        self.g = g          # transverse fields
        self.J = J          # interactions energies
        self.tree = tree    # recursion tree
        self.mix = mix      # mixing pair

    def solve(self):

        # try to read solution from cache
        if self.solver.cache_dir is not None:
            hval, K, hp, inds = hash_problem(h, J, gam=gam)
            hash_pars = {'hval':hval, 'K':K, 'hp':hp, 'inds':inds}
            # try to look up solution
            if hval in self.solver.hash_table:
                try:
                    Es, modes = from_cache(self.solver.cache_dir, **hash_pars)
                except:
