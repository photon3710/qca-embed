#!/usr/bin/env python

#---------------------------------------------------------
# Name: core.py
# Purpose: Common functions for solver methods
# Author:	Jacob Retallick
# Created: 2016.04.11
# Last Modified: 2016.04.11
#---------------------------------------------------------

import numpy as np
import scipy.sparse as sp
from numbers import Number
from collections import Iterable



# Hamiltonian generation methods for basic Ising spin glass

def pauli_z(n, N, pz=[1, -1]):
    '''compute diag(pauli_z) for the n^th of N spins (n in [1...N]). Can 
    override single cell pz if specified'''
    
    if n < 1 or n > N:
        print('Invalid spin index')
        return None
    return np.tile(np.repeat([1, -1], 2**(N-n)), [1, 2**(n-1)])

def generate_Hx(gamma, tril=True):
    '''Fast generation of off-diagonal elements.'''
    
    N = len(gamma)     # number of problem nodes
    Cm = [2**n for n in range(N+1)]
    
    mat = sp.coo_matrix((1,1), dtype=float)
    for n in range(N):
        A = [[mat, None], [gamma[n]*sp.eye(Cm[n], dtype=float), mat]]
        mat = sp.bmat(A)
    
    # compute diagonal of Hamiltonian
    return mat if tril else mat.T
    
def generate_H(h, J, gamma=None):
    '''Generate the sparse Hamiltonian for the given Ising parameters.
    Formatting is assumed to match '''
    
    N = h.size
    N2 = 2**N
    
    # precompute pauli matrices
    SZ = [pauli_z(i, N) for i in range(1, N+1)] # diagonals only
    
    # compute diagonal of Hamiltonian
    H_diag = np.zeros([1,N2], dtype=float)
    for i in range(N):
        H_diag += SZ[i]*h[i]
        for j in range(i+1, N):
            H_diag += J[i, j]*np.multiply(SZ[i], SZ[j])

    if gamma is None:
        return H_diag
    else:
        if isinstance(gamma, Iterable):
            gamma = gamma
        elif isinstance(gamma, Number):
            gamma = [gamma]*N
        Hx = generate_Hx(gamma, tril=True)
        Hx = Hx + sp.tril(Hx, -1).T
        return Hx + sp.diags(H_diag, [0,])

def state_to_pol(state, r=3):
    '''converts a (2^N)xM size state array to an NxM matrix of cell
    polarizations, given to r decimal places.'''
    
    state = np.array(state)
    N = int(np.log2(state.shape[0]))

    amp = np.real(np.multiply(state.conjugate(), state))
    SZ = np.array([pauli_z(i+1, N) for i in range(N)]).reshape([N, -1])

    POL = np.dot(SZ, amp)
    return np.round(POL, r)