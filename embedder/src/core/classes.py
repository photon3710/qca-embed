#!/usr/bin/env python

#---------------------------------------------------------
# Name: classes.py
# Purpose: File constaining functional classes
# Author:	Jacob Retallick
# Created: 12.01.2015
# Last Modified: 12.01.2015
#---------------------------------------------------------

import numpy as np

from core.dense_embed.embed import denseEmbed, setChimera
from core.dense_embed.convert import convertToModels
from core.chimera import tuple_to_linear, linear_to_tuple
import core.core_settings as settings

from dwave_sapi import find_embedding


class Embedding:
    '''Container class for an embedding'''
    
    def __init__(self, qca_file):
        '''Initiate an embedding object'''
        
        # flags and general parameters
        self.qca_file = qca_file    # name of qca file embedded
        self.embed_file = None      # file to save embedding into
        self.coef_dir = None        # directory to save coefficient files into
        self.good = False           # flag for good embedding

        self.full_adj = True    # flag for full adjacency
        self.use_dense = True   # flag for dense placement
        
        self.dense_trials = 1   # number of allowed dense placement trials
        
        # QCA and Chimera structure
        self.qca_adj = {}       # adjacency dict for qca circuit (all cells)
        self.chimera_adj = {}   # adjacency dict for chimera graph
        self.M = None           # number of rows in Chimera graph
        self.N = None           # number of columns in Chimera graph
        self.L = None
        self.active_range = {}  # for mapping between suba dn full graphs
        
        # QCA cell identifications
        self.drivers = set()    # list of driver cells
        self.fixed = set()      # list of fixed cells
        
        # embedding parameters
        self.models = {}        # models for each qca cell

        # export parameters
        self.pols = []          # set of unique polarization to test
        self.J = np.zeros(0)    # array of cell interactions (all cells)
        
    # EMBEDDING METHODS

    def run_embedding(self):
        ''' '''
        
        if self.use_dense:
            self.run_dense_embedding()
        else:
            self.run_heur_embedding()
        
    def run_dense_embedding(self, full_adj=True):
        '''Setup and run the Dense Placement algorithm'''

        # update embedding type in case direct call
        self.use_dense = True
        
        # format embedding parameters
        setChimera(self.chimera_adj, self.M, self.N, self.L)
        active_cells, qca_adj = self.get_reduced_qca_adj()

        # run a number of embedding and choose the best
        embeds = []
        for trial in xrange(settings.DENSE_TRIALS):
            print('Trial {0}...'.format(trial)), 
            try:
                cell_map, paths = denseEmbed(qca_adj, write=False)
                print('success')
            except Exception as e:
                if type(e).__name__ == 'KeyboardInterrupt':
                    raise KeyboardInterrupt
                print('failed')
                continue
            embeds.append((cell_map, paths))
        
        if len(embeds) == 0:
            self.good = False
            return
        
        # sort embedding by number of qubits used (total path length)
        cell_map, paths = sorted(embeds, 
                                 key=lambda x: sum([len(p) for p in x[1]]))[0]
        self.good = True
        
        # get cell models
        models, max_model = convertToModels(paths, cell_map)

        self.models = {k: models[k]['qbits'] for k in models}
    
    def run_heur_embedding(self, full_adj=True):
        '''Setup and run the Heuristic algorithm'''
        
        # update embedding type in case direct call
        self.use_dense = False
        
        active_cells, qca_adj = self.get_reduced_qca_adj()
        S_size = len(qca_adj)
        A_size = len(self.chimera_adj)
        
        # construct S
        S = {}
        for i in range(S_size):
            c1 = active_cells[i]
            for j in range(S_size):
                c2 = active_cells[j]
                v = 1 if c2 in qca_adj[c1] else 0
                S[(i, j)] = v
                S[(j, i)] = v

        # construct A
        A = set()
        for qb1 in self.chimera_adj:
            for qb2 in self.chimera_adj[qb1]:
                l1 = tuple_to_linear(qb1, self.M, self.N, L=self.L, index0=True)
                l2 = tuple_to_linear(qb2, self.M, self.N, L=self.L, index0=True)
                A.add((l1, l2))
 
        try:
            print 'Running heuristic embedding'
            models = find_embedding(S, S_size, A, A_size)
        except Exception as e:
            print(e.message())
        
        print 'Embedding finished'
        self.good = len(models) == S_size
        
        # map models to standard format
        mapper = lambda ind: linear_to_tuple(ind, self.M, self.N, 
                                             L=self.L, index0=True)
        self.models = {active_cells[i]: [mapper(c) for c in models[i]]
            for i in xrange(S_size)}

    # PARAMETER ACCESS
    
    def set_embedder(self, dense=True):
        '''Set embedder type'''
        self.use_dense = dense

    def set_chimera(self, adj, active_range, M, N, L=4):
        '''Set the Chimera graph to embed into'''
        
        self.M = M
        self.N = N
        self.L = L
        self.active_range = active_range
        self.chimera_adj = adj
        
    def set_qca(self, J, cells, full_adj=True):
        '''Set up the qca structure'''
        
        self.full_adj = full_adj

        self.qca_adj = {i: J[i].nonzero()[0].tolist()
            for i in xrange(J.shape[0])}
                
        # driver cells
        self.drivers = set([i for i in self.qca_adj if cells[i].driver])
        
        # fixed cells
        self.fixed = set([i for i in self.qca_adj if cells[i].fixed])
        
    def get_reduced_qca_adj(self):
        '''Get a reduced form of qca_adj only for non-driver/fixed cells'''
        
        # check function for membership in drivers or fixed
        check = lambda cell: cell not in self.drivers.union(self.fixed)
        
        reduced_adj = {c1: [c2 for c2 in self.qca_adj[c1] if check(c2)] 
            for c1 in self.qca_adj if check(c1)}

        return sorted(reduced_adj), reduced_adj
    
    # FILE IO
    
    def write_info_file(self, fname):
        '''Write the information file into the coef directory'''
        pass

    def save_embedding(self):
        '''Save the embedding to file'''
        pass
    
    def save_coefs(self):
        '''Save coefficient files for all unique input sets'''
        pass
        
        