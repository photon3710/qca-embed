#!/usr/bin/env python

#---------------------------------------------------------
# Name: solve.py
# Purpose: Sparse matrix formulation of the QCA solver
# Author:	Jacob Retallick
# Created: 2016.04.11
# Last Modified: 2016.04.11
#---------------------------------------------------------

import numpy as np
import numbers

from time import clock
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

from core import generate_H

# sparse method tollerances
TOL_EIGSH = 1e-7

def solve_sparse(Hs, minimal=False, verbose=False, more=False, exact=False,
                 k = None):
    '''Finds a subset of the eigenstates/eigenvalues for a sparse formatted 
    Hamiltonian'''
    
    N = int(round(np.log2(Hs.shape[0])))    # number of effective cells
    factors = {True: 10, False: 3}
    
    if verbose:
        print('-'*40+'\nEIGSH...\n')
    
    # select number of eigenstates to solve
    
    if isinstance(k, numbers.Number):
        K = k
    else:
        if minimal:
            K = 2
        else:
            K = 1 if N==1 else min(pow(2, N)-1, factors[more]*N)
    
    # force K < Hs size
    K = min(K, Hs.shape[0]-1)

    t = clock()
    
    # run eigsh
    
    try:
        if exact:
            e_vals, e_vecs = eigh(Hs.todense())
        else:
            e_vals, e_vecs = eigsh(Hs, k=K, tol=TOL_EIGSH, which='SA')
    except:
        try:
            e_vals, e_vecs = eigsh(Hs, k=2, tol=TOL_EIGSH, which='SA')
        except:
            if verbose:
                print('Insufficient dim for sparse methods. Running eigh')
            e_vals, e_vecs = eigh(Hs.todense())
    
    if verbose:
        print('Time elapsed (seconds): {0:.3f}'.format(clock()-t))
    
    return e_vals, e_vecs
    
def solve(h, J, gamma=None, minimal=False, verbose=False, more=False, 
          exact=False, k = None, full_output=False):
    
    Hs = generate_H(h, J, gamma=gamma)
    
    e_vals, e_vecs = solve_sparse(Hs, minimal=minimal, verbose=verbose, more=more,
                                  exact=exact, k=k)
    return e_vals, e_vecs
    