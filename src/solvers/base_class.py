#!/usr/bin/env python

import numpy as np
import scipy.sparse as sp

import numbers
import collections


class Solver:
    '''Base class for solvers for finding states of Ising spin glasses'''
    
    def __init__(self, **settings):
        '''Initialise Ising solver object'''
        
        self.settings = settings
    
    def set_settings(self, **settings):
        '''Force new Solver settings'''
        
        self.settings = settings
    
    def set_problem(self, h, J, gamma=None, triu=True):
        '''
        Specify Ising parameters
        
        inputs: h       : Iterable specifying the local fields
                J       : 2d iterable specifying the spin-spin interactions
                gamma   : Transverse field of each spin. If scalar, all spins
                         are assigned the same gamma.
                triu    : Flag for upper or lower triangular format for J
        '''
        
        # assert formating
        try:
            # manditory parameters
            h = np.array(h).reshape([-1,])
            J = np.array(J)
            
            # reduce J to either upper or lower triangular
            tris = {True: np.triu(J, 1), False: np.tril(J, -1)}
            assert np.all(tris[True] == tris[False].T), 'J is not symmetric'
            J = tris[triu] if np.any(tris[triu]) else tris[not triu].T
            
            assert J.shape == (h.size,)*2, 'J and h have different sizes'
                
            if gamma is not None:
                if isinstance(gamma, numbers.Number):
                    gamma = np.ones(h.size)*gamma
                elif isinstance(gamma, collections.Iterable):
                    gamma = np.array(gamma).reshape([-1,])
                    assert gamma.size == h.size, \
                        'gamma is not specificed for all problem nodes'
        except:
            print('Invalid Ising parameters...')
            return None
        
        self.h = h
        self.J = J
        self.gamma = gamma
    
    def set_params(self, **params):
        '''Set Solver parameters'''
        self.params = params
    
    def solve(self):
        '''Virtual method. Do nothing if not implemented.'''
        print('Solver method in inherited Solver class not defined')
        return None
    
    @staticmethod
    def pauli_z(n, N):
        '''compute diag(pauli_z) for the n^th of N spins (n in [1...N])'''
        if n < 1 or n > N:
            print('Invalid spin index')
            return None
        return np.tile(np.repeat([1,-1], 2**(N-n)), [1, 2**(n-1)])
        
    @staticmethod
    def generate_Hx(gamma, triu=True):
        '''Fast generation of off-diagonal elements.'''
        
        N = len(gamma)     # number of problem nodes
        Cm = [2**n for n in range(N+1)]
        
        mat = sp.coo_matrix((1,1), dtype=float)
        for n in range(N):
            A = [[mat, gamma[n]*sp.eye(Cm[n], dtype=float)], [None, mat]]
            mat = sp.bmat(A)
        
        return mat if triu else mat.T
    
    @staticmethod
    def generate_H(h, J, gamma=None, triu=True):
        '''Generate the sparse Hamiltonian for the given Ising parameters.
        Formatting is assumed to match Solver parameters.'''
        
        N = h.size
        N2 = N**2
        
        # precompute pauli matrices
        SZ = [Solver.pauli_z(i, N) for i in range(1, N+1)] # diagonals only
        
        # compute diagonal of Hamiltonian
        DIAG = np.zeros([N2,], dtype=float)
        for i in range(N):
            DIAG += SZ[i]*h[i]
            for j in range(i+1, N):
                DIAG += J[i, j]*np.multiply(SZ[i], SZ[j])
        
        if gamma is None:
            return DIAG
        else:
            Hx = Solver.generate_Hx(gamma, triu=triu)
            return Hx + sp.diags(DIAG, 0)

        
