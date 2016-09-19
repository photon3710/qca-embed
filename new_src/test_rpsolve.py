#!/usr/bin/env python

import numpy as np
from solvers.rp_solve_2 import RP_Solver
from solvers.rp_solve import rp_solve
from solvers.sparse import solve

from time import time
import sys

WIRE = True
TRIALS = 0

def generate_wire(N, gmax=1.):
    ''' '''

    h = [1]+[0]*(N-1)
    J = np.diag([-1]*(N-1),1)+np.diag([-1]*(N-1),-1)
    gam = gmax

    return h, J, gam

def generate_prob(N, d=1., gmax=20.):
    '''Generate a random Ising sping-glass problem of N spins. Connection
    density can be controlled by setting d in [0->1]. The maximum allowed
    tunneling parameter can be set with gmax.'''

    h = 2*np.random.rand(N)-1
    gam = gmax
    J = 2*np.random.rand(N,N)-1
    J = np.triu(J*(np.abs(J)<d), 1)
    J = J+J.T

    return h, J, gam


def main(N, cache_dir=None):
    '''Generate and test a problem with N spins'''

    if WIRE:
        h, J, gam = generate_wire(N)
    else:
        h, J, gam = generate_prob(N)

    t = time()
    if N < 18:
        print('Exact solution...')
        e_vals, e_vecs = solve(h, J, gamma=gam)
        print(e_vals[:2])
    print('Runtime: {0:.3f} (s)'.format(time()-t))

    t = time()
    solver = RP_Solver(h, J, gam, verbose=False, cache_dir=cache_dir)
    solver.solve()

    print('\n'*3)
    print('New RP solution:')
    print(solver.node.Es[:2])
    print('Runtime: {0:.3f} (s)'.format(time()-t))

    msolver = RP_Solver(h, J, gam)
    t = time()
    modes = solver.node.modes
    msolver.mode_solve(modes)

    print('\n'*3)
    print('Mode solved:')
    print(msolver.node.Es[:2])
    print('Runtime: {0:.3f} (s)'.format(time()-t))

    f = lambda x: np.round(x,2)

    print(solver.node.Hx.shape)
    print(msolver.node.Hx.shape)

    Dx = solver.node.Hx - msolver.node.Hx

    print('Dx diffs...')
    for i,j in np.transpose(np.nonzero(Dx)):
        print('\t{0}:{1} :: {2:.3f}'.format(i,j,Dx[i,j]))


    # resolve
    for _ in range(TRIALS):
        t = time()
        nx, nz = solver.ground_fields()

        solver = RP_Solver(h, J, gam, nx=nx, nz=nz, verbose=False, cache_dir=cache_dir)
        solver.solve()

        print('\n'*3)
        print('New RP solution:')
        print(solver.node.Es[:2])
        print('Runtime: {0:.3f} (s)'.format(time()-t))


if __name__ == '__main__':

    try:
        N = int(sys.argv[1])
    except KeyError:
        print('No problem size given')
        N = 3

    try:
        cache_dir = sys.argv[2]
    except:
        cache_dir = None

    main(N, cache_dir=cache_dir)
