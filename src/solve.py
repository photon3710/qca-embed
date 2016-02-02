#!/usr/bin/python

#---------------------------------------------------------
# Name: solve.py
# Purpose: Sparse matrix formulation of the QCA solver
# Author:	Jacob Retallick
# Created: 02.08.2014
# Last Modified: 12.06.2015
#---------------------------------------------------------

import numpy as np
import scipy.sparse as sp
from time import clock
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.linalg import eigh
from auxil import generateHam, stateToPol

# sparse method tollerances
TOL_EIGSH = 1e-7
TOL_LOBPCG = 1e-4

# lobpcg diag offset (can be arbitrarily large)
H_OFFSET = 100


def solveSparse(Hs, run_lobpcg=False, minimal=False, verbose=False,
                more=False, exact=False):
    '''Finds a subset of the eigenstates/eigenvalues for a sparse
    formatted Hamiltonian'''

    sols = {}
    N = int(round(np.log2(Hs.shape[0])))  # number of cells

    ## EIGSH

    if verbose:
        print '-'*40
        print 'EIGSH...\n'

    if minimal:    # only the lowest two states
        K_EIGSH = 2
    elif more:        # can only find ground state for 2x2 system using eigsh
        K_EIGSH = 1 if N == 1 else min(pow(2, N)-1, 11*N)
    else:
        K_EIGSH = 1 if N == 1 else 3*N

    K_EIGSH = min(K_EIGSH, Hs.shape[0]-1)
    t1 = clock()

    # run eigsh

    try:
        if exact:
            e_vals, e_vecs = eigh(Hs.todense())
        else:
            e_vals, e_vecs = eigsh(Hs, k=K_EIGSH, tol=TOL_EIGSH, which='SA')
    except:  # should not happen unless K_EIGSH assignment rewritten
        try:
            e_vals, e_vecs = eigsh(Hs, k=2, tol=TOL_EIGSH, which='SA')
        except:
            if verbose:
                print 'Insufficient dim for sparse methods. Running eigh'
            e_vals, e_vecs = eigh(Hs.todense())

    if verbose:
        print 'Time elapsed (seconds): %f\n' % (clock()-t1)

    sols['eigsh'] = {}
    sols['eigsh']['vals'] = e_vals
    sols['eigsh']['vecs'] = e_vecs

    ###################################################################
    ## RUN LOBPCG

    ### LOBPCG only works for positive definite matrices and hence is only
    ### reliable for the ground state (for which the n-th leading minor
    ### remains positive definite in the cholesky decomposition)

    if run_lobpcg:
        if verbose:
            print '-'*40
            print 'LOBPCG...\n'

        t2 = clock()

        # guess solution
        approx = np.ones([N, 1], dtype=float)

        # offset Hamiltonian to assert positive definite (for ground state)
        Hs = Hs+sp.diags(np.ones([1, N], dtype=float)*H_OFFSET, [0])

        # run lobpcg, remove offset
        e_vals, e_vecs = lobpcg(Hs, approx, tol=TOL_LOBPCG)
        e_vals -= H_OFFSET

        if verbose:
            print 'Time elapsed (seconds): %f\n' % (clock()-t2)

        sols['lobpcg'] = {}
        sols['lobpcg']['vals'] = e_vals
        sols['lobpcg']['vecs'] = e_vecs

    return sols


def solve(h, J, gamma=None, verbose=False, output=True,
          full_output=False, minimal=False, more=False, exact=False):
    ''' takes as input the h and J coefficients (unique to scale) and
    returns the ground state, the first excited state, and a subset of
    the energy spectrum (number of energies determined by sparse.py)'''

    Hs = generateHam(h, J, gamma)
    sols = solveSparse(Hs, verbose=False, minimal=minimal,
                       more=more, exact=exact)

    ground_state = stateToPol(sols['eigsh']['vecs'][:, 0])
    excited_state = stateToPol(sols['eigsh']['vecs'][:, 1])
    N = len(ground_state)

    spectrum = sols['eigsh']['vals']

    if verbose:
        print '\n'*2+'*'*40+'\nPartial Energy Spectrum\n\n'
        print '\n'.join(map(str, spectrum))

        print '\n'*2+'*'*40+'\nPolarizations\n\n'

        for i in xrange(N):
            print '%d: %.4f %.4f' % (i, ground_state[i], excited_state[i])

    if output:
        if full_output:
            return ground_state, excited_state, spectrum, sols
        else:
            return ground_state, excited_state, spectrum
