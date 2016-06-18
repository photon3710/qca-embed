#!/usr/bin/env python

import numpy as np

from solvers.mc_solve import simulated_annealing, pi_qmc
from solvers.rp_solve import rp_solve
import json, sys, os

from collections import defaultdict

import matplotlib.pyplot as plt

def load_json(fname):
    ''' '''
    
    try:
        fp = open(fname, 'r')
        data = json.load(fp)
        fp.close()
    except:
        print('Failed to process file: {0}'.format(fname))
        return
    
    h_ = data['h']
    J_ = data['J']
    qbits = data['qbits']
    N = len(qbits)
    
    h = np.zeros([N,], dtype=float)
    J = np.zeros([N, N], dtype=float)
    
    qbit_map = {qb: i for i, qb in enumerate(qbits)}
    
    for i, v in h_.iteritems():
        h[qbit_map[int(i)]] = v
    
    for i in J_:
        for j, v in J_[i].iteritems():
            J[qbit_map[int(i)], qbit_map[int(j)]] = v
            J[qbit_map[int(j)], qbit_map[int(i)]] = v
    
    return h, J, data['energies'], data['spins']

def load_coef(fname):
    
    try:
        fp = open(fname, 'r')
    except:
        print('Failed to open file: {0}'.format(fname))
        return
    
    fp.readline()
    
    h_ = {}
    J_ = defaultdict(dict)
    
    for line in fp:
        i, j, v = line.split()
        i, j, v = int(i), int(j), float(v)
        if i==j:
            h_[i] = v
        else:
            J_[i][j] = v
            J_[j][i] = v
    
    qbits = sorted(J_.keys())
    N = len(qbits)

    h = np.zeros([N,], dtype=float)
    J = np.zeros([N, N], dtype=float)
    
    qbit_map = {qb: i for i, qb in enumerate(qbits)}

    for i, v in h_.iteritems():
        h[qbit_map[i]] = v
    
    for i in J_:
        for j, v in J_[i].iteritems():
            J[qbit_map[i], qbit_map[j]] = v
            J[qbit_map[j], qbit_map[i]] = v
    
    return h, J

def sa_sched(T0, Tf, nsteps):
    T = T0
    dT = (Tf-T0)*1./(nsteps-1)
    yield T
    for n in range(nsteps-1):
        T += dT
        yield T

def qmc_sched(G0, Gf, E0, Ef, nsteps):
    ''' '''
    
    G, eps = G0, E0
    
    dG = (Gf-G0)*1./(nsteps-1)
    dE = (Ef-E0)*1./(nsteps-1)
    
    yield eps, G
    for n in range(nsteps-1):
        G += dG
        eps += dE
        yield eps, G
    
def main(fname):
    ''' '''
    
    if os.path.splitext(fname)[1] == '.json':
        h, J, E, S = load_json(fname)
    else:
        h, J = load_coef(fname)
    
#    e_vals, e_vecs, modes = rp_solve(h, J, gam=0.001, verbose=False)

#    e_sa, s_sa = simulated_annealing(h, J, sa_sched(5, 1e-3, 100), 10)
    e_qmc, s_qmc = pi_qmc(h, J, qmc_sched(2, 1e-6, 1, 1, 1000), P=40, mcs=10, T=0.05)
    print(e_qmc, s_qmc)
    

if __name__ == '__main__':
    
    try:
        fname = sys.argv[1]
    except:
        print('No file given...')
        sys.exit()
    
    main(fname)
        