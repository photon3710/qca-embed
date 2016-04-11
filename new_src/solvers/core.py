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
    
    return mat if tril else mat.T
    
def generate_H(h, J, gamma=None, tril=True):
    '''Generate the sparse Hamiltonian for the given Ising parameters.
    Formatting is assumed to match '''
    
    N = h.size
    N2 = N**2
    
    # precompute pauli matrices
    SZ = [pauli_z(i, N) for i in range(1, N+1)] # diagonals only
    
    # compute diagonal of Hamiltonian
    H_diag = np.zeros([N2,], dtype=float)
    for i in range(N):
        H_diag += SZ[i]*h[i]
        for j in range(i+1, N):
            H_diag += J[i, j]*np.multiply(SZ[i], SZ[j])
    
    if gamma is None:
        return H_diag
    else:
        Hx = generate_Hx(gamma, tril=tril)
        return Hx + sp.diags(H_diag, 0)

def state_to_pol(state):
    '''converts a 2^N size state to an N size polarization vector'''
    
    state = np.asmatrix(state)

    ## correct state alignment

    a, b = state.shape
    if a < b:
        state = state.transpose()

    amp = np.abs(np.multiply(state.conjugate(), state))

    N = int(np.log2(np.size(state)))
    SZ = [np.asmatrix(pauli_z(i+1, N)) for i in range(N)]

    POL = [-float(SZ[i]*amp) for i in xrange(N)]
    return POL