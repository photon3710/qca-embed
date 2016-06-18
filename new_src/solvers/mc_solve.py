#!/usr/bin/env python


import numpy as np
from collections import deque

from pprint import pprint


def accept(dE, T):
    return dE<=0 or np.exp(-dE/T) > np.random.random()

def compute_energies(A, spins):
    '''Compute energy of all slices for eps=1.'''
    
    E = np.zeros(spins.shape[0])
    
    for i, a in enumerate(A):
        for j, v in a:
            if j==i:
                E += v*spins[:, i]
            else:
                E += .5*v*spins[:, i]*spins[:, j]
    return E

def bf_local_dE(A, spins, i, k):
    
    E0 = compute_energies(A, spins)
    spins[k,i] *= -1
    Ef = compute_energies(A, spins)
    spins[k,i] *= -1
    return Ef-E0

def local_dE(A, spins, i, k):
    '''calculate local change in energy if spin i flipped in slice k'''
    
    E = 0.
    for j, v in A[i]:
        if j==i:
            E += v
        else:
            E += v*spins[k,j]
    return -2*spins[k,i]*E

def inter_dE(spins, i, k):
    '''calculate change in energy if spin i flipped in all slices'''
    
    P = spins.shape[0]
    left = k-1 if k else P-1
    right = k+1 if k<(P-1) else 0
    return -2*spins[k,i]*(spins[left, i]+spins[right, i])

def format_adjacency(h, J):
    '''Construct sparse adjacency list structure from problem coefficients.
    A[i] is a list of pairs (j, v) for coupling v between i and j (i=j gives
    the bias).'''

    N = h.size
    A = [set() for _ in range(N)]
    for i in np.nonzero(h)[0]:
        A[i].add((i, h[i]))
    for x, y in zip(*np.nonzero(J)):
        A[x].add((y, J[x,y]))
        A[y].add((x, J[x,y]))
    return A, N

def pi_qmc(h, J, sched, P=40, mcs=1, T=0.025):
    '''Run path integral QMC for a given Ising spin glass problem'''
    
    PT = P*T
    
    pt_check = PT/np.max(np.abs(J))
    assert pt_check >= 1, 'T too low for initial thermal eq.: {0}'.format(pt_check)
        
    # build adjacency structure
    A, N = format_adjacency(h, J)
    
    # initialise spins
    sa_sched = np.linspace(3, 1e-5, 100)
    e, s = simulated_annealing(h, J, sa_sched, mcs=1)
    print(e)
    spins = np.tile(s, [P,1])
    print(spins.shape)
    energies = compute_energies(A, spins)
    
    order = range(P)    
    for eps, gam in sched:
        J_perp = -.5*PT*np.log(np.tanh(gam/PT))
        for n_mcs in range(mcs):
            
            # compute local changes, one per slice
            np.random.shuffle(order)    # shuffle slice order
            for k in order:
                i = np.random.randint(0, N)
                local_ediff = local_dE(A, spins, i, k)
                inter_ediff = inter_dE(spins, i, k)
                delta_E = eps*local_ediff-J_perp*inter_ediff
                if accept(delta_E, T):
                    spins[k,i] *= -1
                    energies[k] += local_ediff
            
            # compute global change
            i = np.random.randint(0, N)
            local_ediffs = local_dE(A, spins, i, range(P))
            delta_E = eps*np.sum(local_ediffs)
            if accept(delta_E, T):
                spins[:,i] *= -1
                energies += local_ediffs
    
    e_min = np.min(energies)
    k_min = np.argmin(energies)
    return e_min, spins[k_min, :]

def simulated_annealing(h, J, sched, mcs=1):
    '''Run simulated annealing on a given Ising spin glass problem'''

    # build adjacency structure
    A, N = format_adjacency(h, J)
    
    # initialise spins
    spins = 2*(np.random.random([1, N]) < .5)-1
    E = compute_energies(A, spins)[0]
    print('Initial energy: {0:.3f}'.format(E))
    
    store = deque(maxlen=1000)
    store.append(E)
    
    order = range(N)
    for T in sched:
        for n_mcs in range(mcs):
            # shuffle spin order
            np.random.shuffle(order)
            for i in order:
#                bf_local_ediff = bf_local_dE(A, spins, i, 0)[0]
                local_ediff = local_dE(A, spins, i, 0)
                delta_E = local_ediff
                if accept(delta_E, T):
                    spins[0,i] *= -1
                    E += delta_E
            store.append(E)
    
    return E, spins[0]
                
    
    