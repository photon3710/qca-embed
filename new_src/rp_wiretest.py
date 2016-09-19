#!/usr/bin/python
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from itertools import combinations
from time import time

from solvers.rp_solve import rp_solve
from solvers.rp_solve_2 import RP_Solver
from solvers.spectrum import Spectrum
from solvers.sparse import solve


FS = 16
GS = 18

LW = 2
CACHE = './solvers/cache/wire_temp/'

IMG_DIR = './img/'
SAVE = True

SP_MAX = 16
N_STEPS = 100

TRIALS = 5

def anlyt_grnd(N, gammas):
    '''Analytical ground state of an N cell wire'''
    ground = np.zeros(gammas.shape, dtype=float)
    for k in range(1, N+1):
        a = k*np.pi/(N+1)
        ground -= np.sqrt(np.power(np.sin(a), 2) +
                          np.power(gammas-np.cos(a), 2))
    return ground

def anlyt_gaps(N, gammas, n):
    '''Compute the n^th degenerate subset of energy gaps'''

    combs = list(combinations(range(1, N+1), n))
    gap_fncs = []
    for comb in combs:
        gap = np.zeros(gammas.shape, dtype=float)
        for k in comb:
            a = k*np.pi/(N+1)
            gap += 2*np.sqrt(np.power(np.sin(a), 2) +
                             np.power(gammas-np.cos(a), 2))
        gap_fncs.append(gap)
    return gap_fncs


def gen_wire_coefs(N):
    '''Generate h and J parameters for an N cell wire'''

    h = [1]+[0]*(N-1)
    J = -np.diag([1]*(N-1), 1)

    h = np.array(h)
    J += J.T

    return h, J

def wire_spectrum(N, gmin=0.01, gmax=2.):
    '''Calculate spectrum of an N cell wire for a gamma sweep'''

    gammas = np.linspace(gmin, gmax, 20)
    h, J = gen_wire_coefs(N)

    spectrum = []
    times = []
    for i, gamma in enumerate(gammas):
        sys.stdout.write('\r{0:.1f}%'.format((i+1)*100./(gammas.size+1)))
        sys.stdout.flush()
        t = time()
        solver = RP_Solver(h, J, gamma)
        solver.solve()
        e_vals, e_vecs = solver.node.evd()
        # e_vals, e_vecs, modes = rp_solve(h, J, gam=gamma, cache_dir=CACHE)
        times.append(time()-t)
        spectrum.append(e_vals)

    # make square
    L = min(len(s) for s in spectrum)
    spectrum = [s[:L] for s in spectrum]

    spectrum = np.array(spectrum)

    plt.figure('Wire spectrum')

    grnd = anlyt_grnd(N, gammas)
    for n in range(0, int(np.log2(N))):
        gaps = anlyt_gaps(N, gammas, n)
        for gap in gaps:
            plt.plot(gammas, grnd+gap, 'k-', linewidth=2)

    plt.plot(gammas, spectrum, 'x', markersize=7, markeredgewidth=2)
    plt.xlabel('Gamma', fontsize=FS)
    plt.ylabel('Energy', fontsize=FS)
    plt.title('{0} cell wire spectrum'.format(N), fontsize=FS)
    plt.show(block=False)

    plt.figure('Timer')
    plt.plot(gammas, times, 'x', markersize=5, markeredgewidth=1.5)
    plt.xlabel('Gamma', fontsize=FS)
    plt.ylabel('Runtime (s)', fontsize=FS)
    plt.title('{0} cell wire spectrum: runtime'.format(N), fontsize=FS)
    plt.show()


def new_gamma_sweep(N, gmax=10.):
    ''' '''

    rp_times = []
    sp_times = []

    N_steps = 50
    gammas = np.linspace(1e-5,1, N_steps)
    eps = np.linspace(1,1e-5, N_steps)

    h, J = gen_wire_coefs(N)
    for i, (gam,ep) in enumerate(zip(gammas,eps)):

        sys.stdout.write('\r{0:.1f}%: {1}'.format(i*100/N_steps, i))
        sys.stdout.flush()

        t = time()
        for _ in range(TRIALS):
            sys.stdout.write('.')
            sys.stdout.flush()
            solver = RP_Solver(ep*h, ep*J, gam)
            solver.solve()
        rp_times.append((time()-t)/TRIALS)
        print(rp_times[-1])

        if N <= SP_MAX:
            t = time()
            for _ in range(TRIALS):
                sys.stdout.write(',')
                sys.stdout.flush()
                e_vals, e_vecs = solve(ep*h, ep*J, gamma=gam)
            sp_times.append((time()-t)/TRIALS)

    plt.figure('Gamma-Sweep')
    plt.plot(gammas, rp_times, 'g', linewidth=LW)

    if sp_times:
        plt.plot(gammas, sp_times, 'b', linewidth=LW)

    plt.xlabel('s', fontsize=GS)
    plt.ylabel('Run-time (s)', fontsize=FS)


    plt.text(0.28, .6, 'H = s$H_T$ + (1-s) $H_P$', fontsize=20)

    if SAVE:
        plt.savefig(os.path.join(IMG_DIR, 'rp_wire_gamma_{0}.eps'.format(N)),
                    bbox_inches='tight')
    plt.show()

def new_wire_size_sweep(N_max):
    ''' '''

    rp_times = []
    rp_times2 = []
    sp_times = []
    sp_times2 = []

    Ns = np.arange(2, N_max+1)

    for N in Ns:

        sys.stdout.write('\r{0:.1f}%: {1}'.format((N-1)*100/(N_max-1), N))
        sys.stdout.flush()

        h, J = gen_wire_coefs(N)
        # rp solver
        t = time()
        for _ in range(TRIALS):
            sys.stdout.write(',')
            sys.stdout.flush()
            solver = RP_Solver(h, J, 0)
            solver.solve()
        rp_times.append((time()-t)/TRIALS)

        # rp solver with gamma=eps
        t = time()
        for _ in range(TRIALS):
            sys.stdout.write('.')
            sys.stdout.flush()
            solver = RP_Solver(h, J, 1)
            solver.solve()
        rp_times2.append((time()-t)/TRIALS)

        # exact solver
        if N <= SP_MAX:
            t = time()
            for _ in range(TRIALS):
                sys.stdout.write(',')
                sys.stdout.flush()
                e_vals, e_vecs = solve(h, J, gamma=0)
            sp_times.append((time()-t)/TRIALS)

            t = time()
            for _ in range(TRIALS):
                sys.stdout.write('.')
                sys.stdout.flush()
                e_vals, e_vecs = solve(h, J, gamma=1)
            sp_times2.append((time()-t)/TRIALS)

    # plotting
    plt.figure('Run-times')

    plt.plot(Ns, rp_times, 'g', linewidth=LW)
    plt.plot(Ns[:len(sp_times)], sp_times, 'b', linewidth=LW)

    plt.plot(Ns, rp_times2, 'g--', linewidth=LW)
    plt.plot(Ns[:len(sp_times2)], sp_times2, 'b--', linewidth=LW)

    plt.xlabel('Wire length', fontsize=FS)
    plt.ylabel('Run-time (s)', fontsize=FS)
    plt.legend(['RP-Solver', 'Exact: ARPACK'], fontsize=FS)

    if SAVE:
        plt.savefig(os.path.join(IMG_DIR, 'rp_wire_{0}.eps'.format(N_max)),
                    bbox_inches='tight')
    plt.show()

def wire_size_sweep(N_max):
    '''compute ground and first excited band for up to N_max length
    wires'''

    rp_times = {'naive': [], 'local': [], 'global': []}
    sp_times = []
    Ns = np.arange(2, N_max+1, int(np.ceil((N_max-1)*1./40)))
    for N in Ns:
        sys.stdout.write('\r{0:.1f}%: {1}'.format((N-1)*100/(N_max-1), N))
        sys.stdout.flush()
        h, J = gen_wire_coefs(N)
        for k in rp_times:
            if k=='naive':
                chdir = None
            elif k == 'global':
                chdir = CACHE
            elif k == 'local':
                chdir = os.path.join(CACHE, str(N))
            t = time()
            if False:
                e_vals, e_vecs, modes = rp_solve(h, J, gam=0.01, cache_dir=chdir)
            else:
                solver = RP_Solver(h, J, gam=0, cache_dir=chdir)
                solver.solve()
            rp_times[k].append(time()-t)
        if N <= 0:
            t = time()
            e_vals, e_vecs = solve(h, J, gamma=0.1)
            sp_times.append(time()-t)


    plt.figure('Run-times')
    plt.plot(Ns, rp_times['naive'], 'b', linewidth=2)
    plt.plot(Ns, rp_times['local'], 'g', linewidth=2)
    plt.plot(Ns, rp_times['global'], 'r', linewidth=2)
    if sp_times:
        plt.plot(Ns[:len(sp_times)], sp_times, 'g', linewidth=2)
    plt.xlabel('Wire length')
    plt.ylabel('Run-time (s)')
    plt.legend(['Naive', 'Local', 'Global'], fontsize=FS)
    plt.show()

    if SAVE:
        plt.savefig(os.path.join(IMG_DIR, 'rp_wire_{0}.eps'.format(N_max)),
                    bbox_inches='tight')

if __name__ == '__main__':

    try:
        N = int(sys.argv[1])
    except:
        print('No wire size given...')
        sys.exit()

    # wire_spectrum(N)
    # new_gamma_sweep(N)
    new_wire_size_sweep(N)
