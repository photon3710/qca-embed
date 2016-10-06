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
from collections import Iterable, defaultdict
from heapq import heappop, heappush
from time import time


def ind_to_state(ind, N):
    '''Converts a state index to a list of N spins.

    ex: ind_to_state(0, 5) = (-1, -1, -1, -1, -1)
        ind_to_state(2, 4) = (-1, -1, 1, -1)
    '''

    bnr = format(ind, '#0{0}b'.format(N+2))[2::]
    return [2*int(x)-1 for x in bnr]

def diag_solve(H, k=None):
    '''Quick sort method for "diagonalizing" an array representing the
    diagonal of a Hamiltonian'''

    inds = np.argsort(H)

    if k is None:
        k = H.size

    e_vals = H[inds]
    return e_vals, inds

# Hamiltonian generation methods for basic Ising spin glass

def pauli_z(n, N, pz=[1, -1]):
    '''compute diag(pauli_z) for the n^th of N spins (n in [1...N]). Can
    override single cell pz if specified'''

    if n < 1 or n > N:
        print('Invalid spin index')
        return None
    return np.tile(np.repeat([1, -1], 2**(N-n)), 2**(n-1))

def multi_kron_prod(*mats):
    '''Kronecker of multiple sparse matrices'''
    A = mats[-1]
    for i in range(len(mats)-2, -1, -1):
        A = sp.kron(mats[i], A)
    return A

def multi_kron_sum(*mats):
    '''Compute the Kronecker sum of multiple matrices. Special case if all
    diagonal'''

    N = len(mats)

    diag = all(mat.ndim==1 for mat in mats)

    # determine size of each matrix
    Cm = [1]
    for mat in mats:
        Cm.append(Cm[-1]*mat.shape[0])

    if diag:
        S = np.zeros(Cm[-1])

        for i, mat in enumerate(mats):
            S += np.tile(np.repeat(mat, Cm[N]/Cm[i+1]), Cm[i])
    else:
        S = sp.coo_matrix((Cm[-1],Cm[-1]))

        for i, mat in enumerate(mats):
            e1, e2 = sp.eye(Cm[i]), sp.eye(Cm[N]/Cm[i+1])
            S += multi_kron_prod(e1, mat, e2)

    return S

def generate_Hx(gamma, tril=True):
    '''Fast generation of off-diagonal elements of an ising spin-glass.'''

    if gamma is None:
        return None

    N = len(gamma)     # number of problem nodes
    Cm = [2**n for n in range(N+1)]

    mat = sp.coo_matrix((1,1), dtype=float)
    for n in range(N):
        A = [[mat, None], [gamma[n]*sp.eye(Cm[n], dtype=float), mat]]
        mat = sp.bmat(A)

    return mat + mat.T

def generate_Hz(h, J):
    '''Generate an array containing the diagonal elements of an ising spin-glass
    Hamiltonian with given h and J coefficients'''

    N = h.size
    N2 = 2**N

    assert J.shape == (N,N)

    # precompute pauli matrices
    SZ = [pauli_z(i, N) for i in range(1, N+1)] # diagonals only

    Hz = np.zeros([N2,], dtype=float)
    for i in range(N):
        if h[i]:
            Hz += SZ[i]*h[i]
        for j in range(i+1, N):
            if J[i,j]:
                Hz += J[i,j]*SZ[i]*SZ[j]

    return Hz

def generate_Hxx(chi):
    '''Compute a sum of xx spin iteractions (i<j)'''

    N = chi.shape[0]
    px = sp.coo_matrix([[0,1],[1,0]])
    Cm = [2**n for n in range(N+1)]

    Hxx = sp.coo_matrix((Cm[-1], Cm[-1]))
    for i,j in np.transpose(np.triu(chi,1).nonzero()):
        # compute s_x^i s_x^j
        e1, e2, e3 = sp.eye(Cm[i]), sp.eye(Cm[j-i-1]), sp.eye(Cm[N-j-1])
        Hxx += chi[i,j]*multi_kron_prod(e1, px, e2, px, e3)

    return Hxx


def generate_Hxz(Gam):
    '''Compute a sum of xz spin interactions (i!=j)'''

    N = Gam.shape[0]
    px = sp.coo_matrix([[0,1],[1,0]])
    pz = sp.coo_matrix([[1,0],[0,-1]])
    Cm = [2**n for n in range(N+1)]

    Hxz = sp.coo_matrix((Cm[-1], Cm[-1]))
    for i,j in np.transpose(Gam.nonzero()):
        if i==j:
            continue
        # compute s_x^i s_x^j
        if i<j:
            e1, e2, e3 = sp.eye(Cm[i]), sp.eye(Cm[j-i-1]), sp.eye(Cm[N-j-1])
            Hxz += Gam[i,j]*multi_kron_prod(e1, px, e2, pz, e3)
        else:
            e1, e2, e3 = sp.eye(Cm[j]), sp.eye(Cm[i-j-1]), sp.eye(Cm[N-i-1])
            Hxz += Gam[i,j]*multi_kron_prod(e1, pz, e2, px, e3)

    return Hxz

def generate_H(h, J, gamma=None):
    '''Generate the sparse Hamiltonian for the given Ising parameters. If no
    tunneling parameters set, returns the diagonal of H, otherwise returns a
    symmetric sparse Hamiltonian.'''

    h = np.array(h).reshape([-1,])
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


### --------------------------------------------------------------------------
### ISING HASHING

def bfs_order(x, A):
    '''Determine modified bfs order for degrees x and adjacency matrix A'''
    N = len(x)
    adj = [np.nonzero(A[i])[0] for i in range(N)]
    dx = 1./(N*N)
    C = defaultdict(int)
    pq = [min([(y, i) for i, y in enumerate(x)]),]
    visited = [False]*N
    order = []
    while len(order)<N:
        y, n = heappop(pq)
        if visited[n]:
            continue
        order.append(n)
        visited[n] = True
        for n2 in [a for a in adj[n] if not visited[a]]:
            x[n2] -= C[y]*dx
            heappush(pq, (x[n2], n2))
        C[y] += 1
    return order

def hash_mat(M):
    '''Return a hash value for a numpy array'''
    M.flags.writeable = False
    val = hash(M.data)
    M.flags.writeable = True
    return val

def hash_problem(h, J, gam=None, nx=None, nz=None, res=3):
    '''Generate a hash for a given (h,J) pair. Decimal resolution of h and J
    should be specified by res argument.'''

    # regularize formating
    h = np.array(h).reshape([-1,])
    J = np.array(J)

    # assert sizes
    N = h.size
    assert J.shape == (N, N), 'J and h are of different sizes'

    if isinstance(gam, Iterable):
        g = np.array(gam).reshape([-1,])
        assert g.size == h.size, 'h and gamma are of different size'
    elif isinstance(gam, Number):
        g = float(gam)
    elif gam is None:
        g = 0.
    else:
        raise AssertionError, 'Invalid format of gamma'

    # normalise
    K = np.max(np.abs(J))
    h /= K
    g /= K
    J /= K

    # reduce resolution to avoid numerical dependence of alpha-centrality
    h = np.round(h, res)
    g = np.round(g, res)
    J = np.round(J, res)

    # compute alpha centrality coefficients for positive parity
    evals = np.linalg.eigvalsh(J)
    lmax = max(np.abs(evals))
    alpha = .99/lmax

    cent = np.linalg.solve(np.eye(N)-alpha*J, h)
    cent += 1- np.min(cent)

    # enumerate centrality coefficients up to given percent resolution
    val, enum = min(cent), 0

    for c, i in sorted([(c,i) for i,c in enumerate(cent)]):
        f = (c-val)/abs(val)
        if f > 1e-10:
            val, enum = c, enum+1
        cent[i] = enum

    # candidate parities
    s = np.sign(np.sum(h))
    hps = [+1, -1] if s == 0 else [s]

    hash_vals = []
    inds = []

    for hp in hps:
        inds.append(bfs_order(cent*hp, J != 0))
        h_ = ((10**res)*h[inds[-1]]*hp).astype(int)
        J_ = ((10**res)*J[inds[-1],:][:, inds[-1]]).astype(int)
        if isinstance(g, Number):
            g_ = int((10**res)*g)
            hash_vals.append(hash((hash_mat(h_), g_, hash_mat(J_))))
        else:
            g_ = ((10**res)*g[inds[-1]]).astype(int)
            nx_ = None if nx is None else ((10**res)*nx[inds[-1]]).astype(int)
            nz_ = None if nz is None else ((10**res)*nz[inds[-1]]).astype(int)
            h0 = (hash_mat(g_), hash_mat(nx_), hash_mat(nz_))
            hash_vals.append(hash((hash_mat(h_), hash(h0), hash_mat(J_))))

    hval, hp, ind = min(zip(hash_vals, hps, inds))
    return hval, K, hp, ind
