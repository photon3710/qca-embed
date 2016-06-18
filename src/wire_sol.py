#!/usr/bin/env python

#---------------------------------------------------------
# Name: wire-sol.py
# Purpose: Investigation of exact wire solution
# Author: Jacob Retallick
# Created: 2015.10.28
# Last Modified: 2015.10.28
#---------------------------------------------------------

from __future__ import division
import sys

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from itertools import permutations, combinations
from solve import solve
from scipy.optimize import curve_fit

FS = 20     # default font size
GFS = 22
MS = 5
MEW = 2

EPS = 1e-9  # very small value
SAVE = False
SHOW_GAPS = False
THRESH = 0.02


def hyper_fit(x, y):
    '''hyperbola fit'''

    # find extremum, guess parameters
    mid = len(x)//2
    curv = np.sign(-2*y[mid]+y[mid-1]+y[mid+1])
    if curv > 0:
        p0 = [0, y.min(), x[y.argmin()], 1]
    elif curv < 0:
        p0 = [0, y.max(), x[y.argmax()], 1]
    else:
        p0 = [0, y[mid], 0, 0]
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    fit_func = lambda x, y0, a, x0, b: y0 + a*np.sqrt(1+b**2*np.power(x-x0, 2))
    popt, pcov = curve_fit(fit_func, x, y, p0=p0)
    func = lambda x: fit_func(x, *popt)

    return popt, func

def check_fermion_model(Y, h):
    ''' '''
    
    N = len(h)
    if h[0]==0:
        N -= 1
    
    # find first excited states, assume Y[:,0] corresponds to gamma=0
    err = 1e-3
    gap0 = Y[0, :]-Y[0, 0]
    print(gap0)

    inds = np.nonzero(np.abs(gap0-2*h[0]) < err)[0].tolist()
    if h[0] != 1:
        inds += np.nonzero(np.abs(gap0-2) < err)[0].tolist()
    
    
    
    # pull off creation energies
    eps_k = Y[:, inds] - np.tile(Y[:, 0].reshape([-1,1]), [1, N])
    
    grnd = -.5*np.sum(eps_k, axis=1)
    
    # analytical model
    gap_fncs = []
    for n in range(0, N+1):
        for comb in combinations(range(N), n):
            gap_fncs.append(grnd + np.sum(eps_k[:, comb], axis=1))
    
    E = np.array(gap_fncs).T
    
    return E, eps_k
    
    

def analytical_gaps(N, n, gammas):
    '''Compute analytical energy gaps for the n^th degenerate set'''

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


def analytical_ground(N, gammas):
    ground = np.zeros(gammas.shape, dtype=float)
    for k in xrange(1, N+1):
        a = k*np.pi/(N+1)
        ground -= np.sqrt(np.power(np.sin(a), 2) +
                          np.power(gammas-np.cos(a), 2))
    return ground


def wire_J(N):
    return np.diag([-1]*(N-1), 1)+np.diag([-1]*(N-1), -1)


def correct_crossing(M):
    '''Given 3xk matrix all, find an index permutation of 1,.., k that
    minimizes MS of curvature coefficients'''
    k = M.shape[1]
    # split into approximately degenerate sub_problems
    thresh = THRESH*(np.max(M[1])-np.min(M[1]))
    D = np.abs(np.outer(M[1], np.ones(k))-np.outer(np.ones(k), M[1])) <= thresh
    sub_inds = [sorted(x) for x in
                nx.connected_components(nx.Graph(D)) if len(x) > 1]
    out_inds = range(k)
    for inds in sub_inds:
        if len(inds) > 6:
            continue
        # calculate curvature matrix
        A = np.outer(2*M[1, inds]-M[0, inds], np.ones(len(inds))) -\
            np.outer(np.ones(len(inds)), M[2, inds])
        # assign costs to each permutation
        r = range(len(inds))
        costs = [(np.sum(np.abs(A[r, x])), x) for x in permutations(r)]
        # pick smallest cost permutation
        opt_inds = list(sorted(costs)[0][1])
        for i in xrange(len(inds)):
            out_inds[inds[i]] = inds[opt_inds[i]]
    return out_inds


def correct_crossings(M):
    '''Correct for path crossings in columns of a matrix'''

    M = np.array(M)
    N = M.shape[0]

    for i in xrange(1, N-1):
        inds = correct_crossing(M[i-1:i+2, :])
        M[(i+1)::, :] = M[(i+1)::, inds]
    return M


def sweep_gamma(h, J, gmin, gmax, steps=2, show=True):
    '''Sweep gamma values for circuit (h, J) from gmin to gmax in the given
    number of steps.'''

    gammas = np.linspace(gmin, gmax, steps)

    egaps = []
    Es = []
    exact = len(h) < 8
    for i in xrange(len(gammas)):
        gamma = gammas[i]
        sys.stdout.write('\r{0:.1f} %'.format(100*i/len(gammas)))
        sys.stdout.flush()
        gs, es, spec = solve(h, J, gamma=gamma,
                             more=True, minimal=False, exact=exact)
        egaps.append([spec[i]-spec[0] for i in xrange(1, len(spec))])
        Es.append(spec)

    egaps = np.array(egaps)
    Es = np.array(Es)

    Y = egaps if SHOW_GAPS else Es

    # correct for crossings
    Y = correct_crossings(Y)

    if show:
        plt.figure('Gamma sweep: egaps')
        if SHOW_GAPS:
            plt.plot(gammas, Y, 'x', markersize=MS, markeredgewidth=MEW)
            plt.xlabel('$\gamma$', fontsize=GFS)
            plt.ylabel('$E_{gap}$/$J$', fontsize=GFS)
            plt.ylim([0, plt.ylim()[1]])
        else:
            plt.plot(gammas, Y, 'x', markersize=MS, markeredgewidth=MEW)
            plt.xlabel('$\gamma$', fontsize=GFS)
            plt.ylabel('$E$/$J$', fontsize=GFS)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.show(block=False)

    # fitting
#    for i in range(Y.shape[1]):
#        try:
#            popt, func = hyper_fit(gammas, Y[:, i])
#            plt.plot(gammas, func(gammas), '--')
#        except:
#            print('Fit failed...')

    # analytical solutions
    if True:
        E_an, eps_k = check_fermion_model(Y, h)
        plt.plot(gammas, E_an, 'k-', linewidth=2)
    else:
        if SHOW_GAPS:
            ground = np.zeros(gammas.shape, dtype=float)
        else:
            ground = analytical_ground(len(h), gammas)
        for n in xrange(0, N+1):
            gaps = analytical_gaps(len(h), n, gammas)
            for gap in gaps:
                plt.plot(gammas, ground+gap, 'k-', linewidth=2)

    # save figure
    if SAVE:
        plt.savefig('../img/wire_sol_{0}.eps'.format(len(h)),
                    bbox_inches='tight')
    # force show with block
    if show:
        plt.show(block=True)


if __name__ == '__main__':

    try:
        N = int(sys.argv[1])
    except:
        print('No wire length given...')
        sys.exit()

    try:
        gpar = [float(sys.argv[i]) for i in [2, 3, 4]]
    except:
        print 'Insufficient gamma parameters... using defaults'
        gpar = [-5, 5, 60]

    J = wire_J(N)
    h = [.435]+[0]*(N-2) + [.6]
    sweep_gamma(h, J, gpar[0], gpar[1], steps=gpar[2])
