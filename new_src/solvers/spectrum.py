#!/usr/bin/env python

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from numbers import Number

from solvers.rp_solve import rp_solve, build_comp_H
from solvers.rp_solve_2 import RP_Solver
from solvers.sparse import solve_sparse, solve

from scipy.signal import argrelmin
from scipy.optimize import curve_fit
from time import time
from collections import defaultdict

from pprint import pprint
import os, re, sys

FS = 14
CACHE = './solvers/cache/rp_cache/0/'
CACHE2 = './solvers/cache/new_rp_cache/0/'

CACHING = False

GAP_MAX = 2.
EPS_MIN = 1e-5        # effective eps = 0
EXACT_THRESH = 18   # maximum problem size for exact solver
OLD_SOLVER = False

SAVE_FIG = False
MODEL_FILE = './data/models/d_wave_lz.txt'

def prepare_models():
    ''' '''

    try:
        fp = open(MODEL_FILE, 'r')
    except:
        print('Failed to load model file...')
        return None

    opts = {}
    for line in fp:
        data = line.strip().split()
        rt, opt = int(data[0]), [float(x) for x in data[1:]]
        opts[rt] = opt

    model = lambda x, a, b: 1./(1+np.exp((x-b)/abs(a)))
    models = {rt: lambda x: model(x, *opts[rt]) for rt in opts}

    return models

def analyse_crossing(delta, n, dx, show=False):
    ''' Analyse an avoided crossing with given delta energies about an
    approximate crossing location (index)'''

    fit_func = lambda x, x0, d0, w: d0**2*(1+(x-x0)**2/w**2)

    X = np.arange(max(0, n-dx), min(len(delta-1), n+dx))
    Y = delta[X]

    p0 = [n, np.min(Y), dx]
    try:
        popt, pcov = curve_fit(fit_func, X, Y**2, p0=p0)
    except OptimizeWarning as e:
        print(e.message())
        return None, None
    perr = np.sqrt(np.diag(pcov))

    if show:
        xx = np.linspace(0, len(delta-1),100)
        plt.plot(delta, 'x')
        plt.plot(xx, np.sqrt(fit_func(xx, *popt)), '-')
        plt.show()

    return popt, perr

def simplify_params(eqs):
    '''Remove extra variable in equations'''

    # find symbols in each equations
    S = [eq.atoms(sym.Symbol) for eq in eqs]

    symbols = set()
    for s in S:
        symbols = symbols.union(s)

    sym_map = {'p{0}'.format(i): i for i in range(len(symbols))}
    M = np.zeros([len(S), len(symbols)], dtype=int)
    for i, s in enumerate(S):
        for x in s:
            M[i][sym_map[str(x)]]=1

    D = np.sum(M,axis=0)
    inds = np.nonzero(D != 1)[0]
    kills = {'p{0}'.format(x): 0 for x in inds}

    eqs = [eq.subs(kills) for eq in eqs]

    return eqs, inds


def simplify_eqs(eqs):
    '''Remove as many extra equations as possible'''

    print('Simplifying equations...')

    # find symbols in each equations
    S = [eq.atoms(sym.Symbol) for eq in eqs]

    symbols = set()
    for s in S:
        symbols = symbols.union(s)

    sym_map = {str(x): i for i, x in enumerate(symbols)}
    M = np.zeros([len(S), len(symbols)], dtype=bool)
    for i, s in enumerate(S):
        for x in s:
            M[i][sym_map[str(x)]]=1

    keep = [True]*M.shape[0]
    while sum(keep)>M.shape[1]:
        # identify banned indices
        inds = (M.sum(axis=0)==1).nonzero()[0].tolist()
        # remove first equation which does not include a banned index
        for i in range(M.shape[0]-1,-1,-1):
            if not keep[i]:
                continue
            for j in inds:
                if M[i][j]:
                    break
            else:
                keep[i] = False
                M[i] *= False
                break
        else:
            break

    inds = M.sum(axis=1).nonzero()[0]
    print(M[inds,:]*1.)
    eqs = [eq for i, eq in enumerate(eqs) if keep[i]]
    return eqs


class Spectrum:
    '''Class for finding and storing spectra'''

    def __init__(self):
        ''' '''

        self.models = prepare_models()

    def run_exact(self):
        '''Compute the spectrum using an exact solver'''

        print('\nRunning Exact Solver method...')
        t = time()
        spectrum = []
        for i, (gamma, ep) in enumerate(zip(self.gammas, self.eps)):
            sys.stdout.write('\r{0:.2f}%'.format(i*100./self.nsteps))
            sys.stdout.flush()
            e_vals, e_vecs = solve(ep*self.h, ep*self.J, gamma=gamma,
                                    more=False)
            spectrum.append(e_vals)
        return spectrum

    def run_rp(self, gset, caching, cache_dir):
        '''Compute the spectrum using the old RP-Solver'''

        print('\nRunning Old RP-Solver method...')
        t = time()

        if caching:
            cache_dir = CACHE if cache_dir is None else cache_dir
        else:
            cache_dir = None

        # find initial modes
        print('Finding initial modes...'),
        e_vals, e_vac, modes = rp_solve(self.h, self.J, gam=gset,
                                        verbose=False, cache_dir=cache_dir)

        print('{0:.2f} sec'.format(time()-t))

        print('Number of included modes: {0}'.format(len(modes)))

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

        for i, (gamma, ep) in enumerate(zip(self.gammas, self.eps)):
            sys.stdout.write('\r{0:.2f}%'.format(i*100./self.nsteps))
            sys.stdout.flush()
            Hs = Hx*gamma
            Hs.setdiag(ep*diag)
            e_vals, e_vecs = solve_sparse(Hs, more=True)
            spectrum.append(e_vals)
        print('...done')

        return spectrum

    def run_rp2(self, rp_steps, caching, cache_dir):
        '''Compute the spectrum using the new RP-Solver'''

        print('\nRunning New RP-Solver method...')
        t = time()

        if caching:
            cache_dir = CACHE2 if cache_dir is None else cache_dir
        else:
            cache_dir = None

        spectrum = []
        for i, (gam, ep) in enumerate(zip(self.gammas, self.eps)):
            sys.stdout.write('\r{0:.2f}%'.format(i*100./self.nsteps))
            sys.stdout.flush()
            ep = max(ep, EPS_MIN)
            solver = RP_Solver(ep*self.h, ep*self.J, gam,
                                cache_dir=cache_dir)
            solver.solve()
            e_vals, e_vecs = solver.node.evd()
            spectrum.append(list(e_vals))

        return spectrum

        # split schedule into iso-field steps
        rp_steps = max(1, rp_steps)
        niso = int(np.ceil(len(self.gammas)*1./rp_steps))

        isos = []
        for i in range(rp_steps):
            gams = self.gammas[i*niso:(i+1)*niso]
            eps = self.eps[i*niso:(i+1)*niso]
            isos.append((gams, eps))

        # solve spectrum within each iso-step
        spectrum = []
        i = 0
        for gammas, eps in isos:
            i += 1
            # solve first problem is iso-step
            gam, ep = gammas[0], max(eps[0], EPS_MIN)
            solver = RP_Solver(ep*self.h, ep*self.J, gam,
                                cache_dir=cache_dir)
            solver.solve()
            e_vals, e_vecs = solver.node.evd()
            spectrum.append(list(e_vals))

            nx, nz = solver.get_current_fields()
            modes = solver.node.modes

            # solve remaining problems is iso-step
            for gam, ep in zip(gammas[1:], eps[1:]):
                i += 1
                sys.stdout.write('\r{0:.2f}%'.format(i*100./self.nsteps))
                sys.stdout.flush()
                ep = max(ep, EPS_MIN)
                solver = RP_Solver(ep*self.h, ep*self.J, gam,
                                    nx=nx, nz=nz, cache_dir=cache_dir)
                solver.mode_solve(modes)
                spectrum.append(list(solver.node.e_vals))

        return spectrum


    def solve(self, h, J, eps, gammas, show=True, gset=0.5,
                rp_steps=10, exact=False, caching=CACHING, cache_dir=None):
        ''' '''

        self.h = np.array(h).reshape([-1,])
        self.J = np.array(J)
        self.eps = eps
        self.gammas = gammas

        self.nsteps = len(gammas)
        self.s = np.linspace(0, 1, self.nsteps)

        print('Problem size: {0}...'.format(len(h)))
        if exact and len(h)>EXACT_THRESH:
            print('Problem too large for exact solver...')
            exact=False

        # exact solver
        if exact:
            spectrum = self.run_exact()

        # old rp-solver
        elif OLD_SOLVER:
            spectrum = self.run_rp(gset, caching, cache_dir)

        else:
            spectrum = self.run_rp2(rp_steps, caching, cache_dir)

        N = min(len(x) for x in spectrum)
        spectrum = [spec[:N] for spec in spectrum]

        self.spectrum = np.array(spectrum)
        Y = self.spectrum - self.spectrum[:,0].reshape([-1,1])
        if show:
            plt.figure('Spectrum')
            plt.plot(self.s, Y, 'x')
            plt.xlabel('Affine parameter (s)', fontsize=FS)
            plt.ylabel('Energy', fontsize=FS)
            # plt.ylim([0,2])
            if SAVE_FIG:
                plt.savefig('./img/spectrum.eps', bboxinches='tight')
            plt.show(block=True)

    def ground_check(self, occ, show=False):

        try:
            assert hasattr(self, 's') and hasattr(self, 'spectrum')
        except AssertionError:
            print('Spectrum has not yet been solved...')
            return None

        delta = self.spectrum[:,1]-self.spectrum[:,0]
        xs = argrelmin(delta)[0]

        # accept the minimum gap and all sufficients small gaps
        dmin = min(delta[x] for x in xs)

        if show:
            plt.plot(delta)
            plt.show()
        xs = [x for x in xs if delta[x] < GAP_MAX or delta[x]<1.1*dmin]
        if len(xs) != 1:
            return [None]

        # only one gap, extract prob
        dx = int(self.nsteps*.05)
        popt, perr = analyse_crossing(delta, xs[0], dx, show=show)

        print(popt)

        if popt is None:
            return [None,]

        prob = occ[0]*1./np.sum(occ)

        params = [(prob, popt[1], abs(popt[2])/self.nsteps)]
        return params

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
        gaps = [self.deltas[n][x] for x, n in xings]

        # collect degenerate energy bins
        self.rp_bins = defaultdict(list)
        spectrum = np.round(self.spectrum[-1,:], 3)

        # causal filter
        self.n_active = 0
        causal_xings = []
        for (x, n), gap in zip(xings, gaps):
            if n <= self.n_active and gap < GAP_MAX:
                causal_xings.append((x, n))
                if n == self.n_active:
                    self.n_active += 1

        self.causal_xings = []
        self.xing_params = []
        if show:
            plt.figure('Xings')
        dx = int(self.nsteps*.05)
        for x, n in causal_xings:
            try:
                popt, perr = analyse_crossing(self.deltas[n], x, dx, show=show)
                print(x, n, popt[1], popt[2]/self.nsteps)
                if popt[1] > GAP_MAX:
                    raise Exception
                self.causal_xings.append((x,n, popt[1], abs(popt[2]/self.nsteps)))
            except Exception:
                continue

        if show:
            plt.show(block=True)

        return len(self.causal_xings)

    def compute_eqs(self):

        # formulate output equations
        self.P = sym.symbols(['p{0}'.format(x) for x in range(len(self.causal_xings))])
        self.eqs = [1]+[0]*(self.n_active)

        rp_bins = defaultdict(list)
        spectrum = np.round(self.spectrum[-1,:], 3)

        E = spectrum[0]
        for i, e in enumerate(spectrum):
            if abs(E-e) > 1e-3*abs(E):
                E = e
            rp_bins[E].append(i)

        t = [0,0]
        for i, (x, n, g, w) in enumerate(self.causal_xings):
            t[0] = self.P[i]*self.eqs[n]+(1-self.P[i])*self.eqs[n+1]
            t[1] = (1-self.P[i])*self.eqs[n]+self.P[i]*self.eqs[n+1]
            self.eqs[n], self.eqs[n+1] = t[0], t[1]

        # sum equations in each energy bin
        e_eqs = {}
        for e in sorted(rp_bins):
            e = round(e, 5)
            e_eqs[e]=0
            for i in rp_bins[e]:
                if i < len(self.eqs):
                    e_eqs[e] += self.eqs[i]
            if e_eqs[e]==0:
                e_eqs.pop(e)

        energies = sorted(e_eqs)
        eqs = [e_eqs[e] for e in energies]
        eqs, inds = simplify_params(eqs)
        e_eqs = {e: eqs[i] for i, e in enumerate(energies)}
        self.e_eqs = e_eqs
        self.P = [self.P[i] for i in inds]
        print('start')
        pprint(self.e_eqs)
        print(self.P)
        print('end')
        return e_eqs

    def predict_probs(self, rt):
        ''' '''

        if self.models is None:
            print('No models specified...')
            return None

        if not hasattr(self, 'e_eqs'):
            self.compute_eqs()

        vals = {}
        for p in self.P:
            n = int(str(p)[1:])
            d0, w0 = self.causal_xings[n][2:]
            vals[p] = 1-self.models[rt](d0)

        pr = {e: eq.subs(vals) for e, eq in self.e_eqs.iteritems()}

        return pr

    def format_probs(self, probs, energies):
        ''' '''

        if not hasattr(self, 'e_eqs'):
            self.compute_eqs()

        p = np.array(probs, dtype=float)
        p /= np.sum(p)

        # bin the energies of both solvers
        dw_bins = defaultdict(list)

        E = energies[0]
        for i, e in enumerate(energies):
            if abs(E-e) > 1e-3*abs(E):
                E = e
            dw_bins[E].append(i)

        # sum probs in each energy bin matching e_eqs
        e_probs = {}
        for e1 in sorted(self.e_eqs):
            e_probs[e1]=0
            for e2, inds in dw_bins.items():
                if abs(e1-e2) < 1e-4*max(abs(e1),abs(e2)):
                    for i in inds:
                        e_probs[e1] += round(p[i],5)

        # lump remaining outcomes onto highest energy outcomes
        e_probs[max(self.e_eqs)] += 1-sum(e_probs.values())

        return e_probs


    def pull_causal(self, probs, energies):
        ''' '''

        p = np.array(probs, dtype=float)
        p /= np.sum(p)

        # bin the energies of both solvers
        dw_bins = defaultdict(list)

        E = energies[0]
        for i, e in enumerate(energies):
            if abs(E-e) > 1e-3*abs(E):
                E = e
            dw_bins[E].append(i)

        e_eqs = self.compute_eqs()
        e_max = max(e_eqs)

        # sum probs in each energy bin matching e_eqs
        e_probs = {}
        for e1 in sorted(e_eqs):
            e_probs[e1]=0
            for e2, inds in dw_bins.items():
                if abs(e1-e2) < 1e-4*max(abs(e1),abs(e2)):
                    for i in inds:
                        e_probs[e1] += round(p[i],5)

        # lump remaining outcomes onto highest energy outcomes
        e_probs[max(e_eqs)] += 1-sum(e_probs.values())

        # generate list of equations to solve
        eqs = [e_eqs[E]-e_probs[E] for E in e_eqs]
        eqs = simplify_eqs(eqs)

        if len(eqs)>4:
            return []

        if False:
            pprint(e_eqs)
            pprint(e_probs)
            pprint(eqs)
            print(self.P)

        print(eqs)
        try:
            out = sym.solve(eqs, self.P, force=True)
        except Exception as e:
            print(e.message)
            out = []

        print('out: {0}'.format(str(out)))

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
        outs = [None]*len(self.causal_xings)
        for k, v in params.items():
            i = int(str(k)[1::])
            outs[i] = [v]+list(self.causal_xings[i][2::])
        print(outs)
        return outs
