#!/usr/bin/env python

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from numbers import Number

from solvers.rp_solve import rp_solve, build_comp_H
from solvers.sparse import solve_sparse
from scipy.signal import argrelmin
from scipy.optimize import curve_fit
from time import time
from collections import defaultdict

from pprint import pprint
import os, re, sys

FS = 14
CACHE = './solvers/cache/rp_cache/0/'

def analyse_crossing(delta, n, dx, show=False):
    ''' Analyse an avoided crossing with given delta energies about an
    approximate crossing location (index)'''

    fit_func = lambda x, x0, d0, w: d0**2+(x-x0)**2/w**2

    X = np.arange(max(0, n-dx), min(len(delta-1), n+dx))
    Y = delta[X]

    p0 = [n, np.min(Y), dx]
    popt, pcov = curve_fit(fit_func, X, Y**2, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    if show:
        xx = np.linspace(0, len(delta-1),100)
        plt.plot(delta, 'x')
        plt.plot(xx, np.sqrt(fit_func(xx, *popt)), '-')

    return popt, perr


class Spectrum:
    ''' '''

    def __init__(self):
        ''' '''
        pass

    def solve(self, h, J, eps, gammas, show=True):
        ''' '''

        self.h = np.array(h).reshape([-1,])
        self.J = np.array(J)
        self.eps = eps
        self.gammas = gammas

        self.nsteps = len(gammas)
        self.s = np.linspace(0, 1, self.nsteps)

        # find initial modes
        print('Finding initial modes...'),
        t = time()
        e_vals, e_vac, modes = rp_solve(self.h, self.J, gam=0.5, verbose=False,
                                        cache_dir=CACHE)

        print('{0:.2f} sec'.format(time()-t))

        # pre-compute sparse Hamiltonian components
        print('Pre-computing sparse Hamiltonian components...'),
        t = time()
        Hx = build_comp_H([self.h], [self.J], [], 1., [modes])
        diag = Hx.diagonal()
        Hx.setdiag([0]*diag.size)
        print('{0:.2f} sec'.format(time()-t))

        spectrum = []

        print('number of modes: {0}'.format(len(modes)))
        print('Estimating spectrum sweep...')

        for i, (gamma, ep) in enumerate(zip(gammas, eps)):
            sys.stdout.write('\r{0:.2f}%'.format(i*100./self.nsteps))
            sys.stdout.flush()
            Hs = Hx*gamma
            Hs.setdiag(ep*diag)
            e_vals, e_vecs = solve_sparse(Hs, more=True)
            spectrum.append(e_vals)
    #        e_vals, e_vecs, m = rp_solve(h, J, gam=gamma)
    #        rp_spectrum.append(e_vals)
        print('...done')

        N = min(len(x) for x in spectrum)
    #    N = min(N, min(len(x) for x in rp_spectrum))
        spectrum = [spec[:N] for spec in spectrum]
    #    rp_spectrum = [spec[:N] for spec in rp_spectrum]

        self.spectrum = np.array(spectrum)
        Y = self.spectrum - self.spectrum[:,0].reshape([-1,1])
        if show:
            plt.figure('Spectrum')
            plt.plot(self.s, Y, 'x')
            plt.xlabel('Affine parameter (s)', fontsize=FS)
            plt.ylabel('Energy', fontsize=FS)
            plt.show(block=False)

    def build_causal(self, show=False):
        ''' '''

        try:
            assert hasattr(self, 's') and hasattr(self, 'spectrum')
        except AssertionError:
            print('Spectrum has not yet been solved...')
            return None

        N = self.spectrum.shape[1]   # number of traced states
        deltas = [self.spectrum[:,k+1]-self.spectrum[:,k] for k in range(N-1)]
        self.deltas = np.array(deltas)

        inds, xs = argrelmin(self.deltas, axis=1, order=1)
        xings = sorted(zip(xs, inds), key=lambda x: (x[0], -x[1]))

        # collect degenerate energy bins
        self.rp_bins = defaultdict(list)
        spectrum = np.round(self.spectrum[-1,:], 3)

        # causal filter
        self.n_active = 0
        self.causal_xings = []
        for x, n in xings:
            if n <= self.n_active:
                self.causal_xings.append((x, n))
                if n == self.n_active:
                    self.n_active += 1

        self.xing_params = []
        if show:
            plt.figure('Xings')
        dx = int(self.nsteps*.05)
        for x, n in self.causal_xings:
            try:
                popt, perr = analyse_crossing(self.deltas[n], x, dx, show=show)
                self.xing_params.append([popt[1], popt[2]/self.nsteps])
            except:
                self.xing_params.append(None)
        if show:
            plt.show(block=True)


    def pull_causal(self, probs, energies):
        ''' '''

        p = np.array(probs, dtype=float)
        p /= np.sum(p)

        # bin the energies of both solvers
        rp_bins = defaultdict(list)
        dw_bins = defaultdict(list)

        spectrum = np.round(self.spectrum[-1,:], 3)

        E = spectrum[0]
        for i, e in enumerate(spectrum):
            if abs(E-e) > 1e-3*abs(E):
                E = e
            rp_bins[E].append(i)

        E = energies[0]
        for i, e in enumerate(energies):
            if abs(E-e) > 1e-3*abs(E):
                E = e
            dw_bins[E].append(i)

        # formulate output equations
        self.P = sym.symbols(['p{0}'.format(x) for x in range(len(self.causal_xings))])
        self.eqs = [1]+[0]*(self.n_active)

        t = [0,0]
        for i, (x, n) in enumerate(self.causal_xings):
            t[0] = self.P[i]*self.eqs[n]+(1-self.P[i])*self.eqs[n+1]
            t[1] = (1-self.P[i])*self.eqs[n]+self.P[i]*self.eqs[n+1]
            self.eqs[n], self.eqs[n+1] = t[0], t[1]

        # sum equations in each energy bin
        e_eqs = {}
        for e in sorted(rp_bins):
            e_eqs[e]=0
            for i in rp_bins[e]:
                if i < len(self.eqs):
                    e_eqs[e] += self.eqs[i]
            if e_eqs[e]==0:
                e_eqs.pop(e)

        # sum probs in each energy bin matching e_eqs
        e_probs = {}
        for e1 in sorted(e_eqs):
            e_probs[e1]=0
            for e2, inds in dw_bins.items():
                if abs(e1-e2) < 1e-4*max(abs(e1),abs(e2)):
                    for i in inds:
                        e_probs[e1] += p[i]

        if len(e_eqs)>4 or len(self.P)>4:
            return [], 1.

        # renormalize probs
        p_fact = np.sum(e_probs.values())
        e_probs = {k: round(v/p_fact, 5) for k, v in e_probs.items()}

        print('p-fact: {0:.3f}'.format(p_fact))

        # generate list of equations to solve
        eqs = [e_eqs[E]-e_probs[E] for E in e_eqs]

        if False:
            pprint(e_eqs)
            pprint(e_probs)
            pprint(eqs)

        try:
            out = sym.solve(eqs, self.P, force=True)
        except:
            out = []

        # get solved parameters
        params = {}
        if out:
            print('solutions found')
            if isinstance(out, list):
                out = out[0]
            for k in out:
                try:
                    params[k] = float(out[k])
                except:
                    pass

        # combine with xing parameters
        outs = [None]*len(self.xing_params)
        for k, v in params.items():
            i = int(str(k)[1::])
            if self.xing_params[i] is not None:
                outs[i] = [v]+self.xing_params[i]
        print(outs)
        return outs, p_fact
