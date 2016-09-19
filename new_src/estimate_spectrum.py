#!/usr/bin/env python

from parse_qca import parse_qca_file
from auxil import CELL_FUNCTIONS, gen_pols, convert_adjacency
from solvers.rp_solve import rp_solve, build_comp_H, compute_rp_tree
from solvers.sparse import solve_sparse, solve
from solvers.spectrum import Spectrum
from scipy.signal import argrelmin

import numpy as np
import matplotlib.pyplot as plt
import sys, os, re, json

from collections import defaultdict
from time import time

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


ADJ = 'adj'
CACHE = './solvers/cache/rp_cache/3/'

FULL_RP = False      # flag for running rp-solver at each gamma
EXACT = True       # flag for running exact solver at each gamma

ZERO = True
EXACT_THRESH = 15   # max circuit size to run the exact solver on

USE_SCHED = True
sched_file = '../dat/schedules/Sys4.txt'

IMG_DIR = './img/'
SAVE = True

FS = 16     # fontsize


def load_schedule(fname):
    ''' '''

    try:
        fp = open(fname, 'r')
    except IOError:
        print('Failed to open schedule file...')
        return

    # re for matching numbers in scientific notation
    regex = re.compile('[+\-]?\d*\.?\d+[eE][+\-]\d+')

    S, Delta, Eps = [], [], []

    for line in fp:
        a, b, c = [float(x) for x in regex.findall(line)]
        S.append(a)
        Delta.append(b/2)
        Eps.append(c)

    fp.close()

    return S, Delta, Eps


def query_schedule(fname, s):
    '''Get the values of the schedule at the parameters s'''

    S, Delta, Eps = load_schedule(fname)

    assert np.min(s) >= 0 and np.max(s)<=1, 'Invalid range for s...'

    delta = np.interp(s, S, Delta)
    eps = np.interp(s, S, Eps)

    return delta, eps

def show_tree(J, tree):
    ''' '''

    G = nx.Graph(J)
    pos = graphviz_layout(G)

    if not tree['children']:
        return

    # color nodes
    colors = ['r', 'b', 'g', 'k', 'm', 'c', 'o', 'y']
    for i, child in enumerate(tree['children']):
        inds = child['inds']
        print(inds)
        color = colors[i%len(colors)]
        nx.draw_networkx_nodes(G, pos, nodelist=inds, node_color=color)
    nx.draw_networkx_edges(G, pos)
    plt.show()

    for child in tree['children']:
        inds = child['inds']
        show_tree(J[inds, :][:, inds], child)

def refine_gap_symm(delta, n):
    '''Refine hap estimate assuming symmetric spacing'''
    d = (delta[n+1]-delta[n-1])*1./(delta[n-1]-2*delta[n]+delta[n+1])
    return n-d/2, delta[n] - (delta[n+1]-delta[n-1])*d/8

def refine_gap(delta, n):
    '''Refine gap estimate given energy differences and guess location'''

    # assume gap is in the interval [n-1, n+1].
    # refine using a quadratic spline

    X = [n-1, n, n+1]
    Y = [delta[x] for x in X]

    denom = (X[1]-X[0])*(X[2]-X[1])*(X[0]-X[2])
    a = (X[2]*(Y[1]-Y[0])+X[1]*(Y[0]-Y[2])+X[0]*(Y[2]-Y[1]))*1./denom
    b = -((Y[1]-Y[0])*X[2]**2 + (Y[0]-Y[2])*X[1]**2 + (Y[2]-Y[1])*X[0]**2)*1./denom
    c = -((X[1]-X[0])*X[0]*X[1]*Y[2]+(X[2]-X[1])*Y[0]*X[1]*X[2]+(X[0]-X[2])*X[0]*Y[1]*X[2])*1./denom

    loc = -b/(2*a)
    gap = c-b*b/(4*a)

    return loc, gap

def integ_lims(delta, n):
    '''determine integration limits about a local minimum'''

    # left limit
    left = n-1
    while left > 0:
        if delta[left-1]>delta[left]:
            left -= 1
        else:
            break

    # right limit
    right = n+1
    while right < len(delta)-1:
        if delta[right+1]>delta[right]:
            right += 1
        else:
            break

    return left, right


def analyse_crossing(delta, m, ta=20e-6):
    '''Analyse avoided crossing given the energy difference, approximate
    location of crossing, and total annealing time'''

    dt = ta/len(delta)

    # improve minimum gap estimate
    loc, gap = refine_gap(delta, m)

    # find integration limits
    left, right = integ_lims(delta, m)

    # compute delta derivative and
    ddelt = (delta[left+2:right+1]-delta[left:right-1])/(2*dt)
    delt = delta[left+1:right]

    # known parameters
    alpha = .25*delt**2
    beta = .5*delt*np.abs(ddelt)/np.sqrt(delt**2-gap**2)
    A = 12+dt*dt*alpha
    B = 12-5*dt*dt*alpha

    # initialize queues
    X = [1, 1-.5*alpha[0]*dt**2]
    Y = [0, .5*dt*(np.sqrt(delt[0]**2-gap**2)-beta[0]*dt)]

    # evolution
    for n in range(1, len(delt)-1):
        x = (2*B[n]-10*(dt**4)*beta[n]*beta[n+1]/A[n+1])*X[1] -\
            (A[n-1]+(dt**4)*beta[n-1]*beta[n+1]/A[n+1])*X[0] +\
            (10*beta[n]+2*B[n]*beta[n+1]/A[n+1])*(dt**2)*Y[1] +\
            (beta[n-1]-A[n-1]*beta[n+1]/A[n+1])*(dt**2)*Y[0]
        x /= A[n+1]*(1+(beta[n+1]*(dt**2)/A[n+1])**2)
        y = (2*B[n]*Y[1]-A[n-1]*Y[0]-(dt**2)*(beta[n+1]*x+10*beta[n]*X[1]+beta[n-1]*X[0]))/A[n+1]
        X = [X[1], x]
        Y = [Y[1], y]
        sys.stdout.write('\r{0:.4f}'.format(abs(x)**2+abs(y)**2))
        sys.stdout.flush()

    print('\n{0:.1f}: {1:.2e}'.format(loc, gap))
    plt.figure('Avoided crossing')
    plt.plot(delta, 'x')
    plt.axvline(loc)
    plt.show()

def locate_crossings(s, spectrum):
    ''' '''

    N = spectrum.shape[1]   # number of traced states
    deltas = [spectrum[:,k+1]-spectrum[:,k] for k in range(N-1)]
    deltas = np.array(deltas)

    print(deltas.shape)
    inds, xs = argrelmin(deltas, axis=1, order=1)
    xings = sorted(zip(xs, inds), key=lambda x: (x[0], -x[1]))

    # causal filter
    n_active = 0
    causal_xings = []
    for x, n in xings:
        if n <= n_active:
            causal_xings.append((x, n))
            if n == n_active:
                n_active += 1

    print(causal_xings)

    for x, n in causal_xings:
        analyse_crossing(deltas[n,:], x)


    plt.figure('Deltas')
    plt.plot(s, deltas.T[:,:5], 'x')
    plt.show()


def compare_spectra(est, true):
    '''Quantify the difference between an estimated spectrum and a more
    accurate spectrum'''

    if not hasattr(est, 'deltas'):
        est.build_causal()

    if not hasattr(true, 'deltas'):
        true.build_causal()

    N = min(est.deltas.shape[0], true.deltas.shape[0])
    # compute delta metrics
    D1 = est.deltas[:N, :]
    D2 = true.deltas[:N, :]
    DD = np.abs(D1-D2)


    plt.figure('Deltas')

    plt.plot(DD.T, 'o', markersize=5)

    plt.show(block=True)

def find_true_spectrum(h, J, gmin=0, gmax=1., emin=0., emax=1., nsteps=100):
    ''' '''
    gmin = 1e-5
    gammas = np.linspace(gmax, gmin, nsteps)    # tranverse field scales
    eps = np.linspace(emin, emax, nsteps)          # ising problem scales

    s = np.linspace(0, 1, nsteps)   # annealing schedule parameter

    if len(h) > EXACT_THRESH:
        return

    true_spectrum = Spectrum()
    true_spectrum.solve(h, J, eps, gammas, exact=True, show=False)
    true_spec = true_spectrum.spectrum

    true_spec -= true_spec[:,0].reshape([-1,1])

    plt.plot(s, true_spec, 'x')
    plt.xlabel('Annealing parameter (s)', fontsize=FS)
    plt.ylabel('Energy', fontsize=FS)
    plt.show()


def estimate_spectrum(h, J, gmin=0., gmax=1., emin=0., emax=1., nsteps=10):
    ''' '''
    gmin = 1e-5
    emin = 1e-5
    gammas = np.linspace(gmax, gmin, nsteps)    # tranverse field scales
    eps = np.linspace(emin, emax, nsteps)          # ising problem scales

    s = np.linspace(0, 1, nsteps)   # annealing schedule parameter
    gammas, eps = query_schedule(sched_file, s)

    # show recursive tree
    if False:
        tree = compute_rp_tree(J)
        show_tree(J, tree)

    # check if solvable using exact methods
    EXACT_check = EXACT and len(h) < EXACT_THRESH

    approx_spectrum = Spectrum()
    rp_spectrum = Spectrum()
    true_spectrum = Spectrum()

    # find approximate spectrum
    print('\nFinding approximate spectrum...')
    approx_spectrum.solve(h, J, eps, gammas, gset=0.5, rp_steps=10, show=False)
    approx_spec = approx_spectrum.spectrum

    if FULL_RP:
        # find more accurate approximate spectrum
        print('\nFinding more accurate spectrum...')
        rp_spectrum.solve(h, J, eps, gammas, gset=2.0, rp_steps=20, show=False)
        rp_spec = rp_spectrum.spectrum

    if EXACT_check:
        # find exact solution
        print('\nFinding true spectrum...')
        true_spectrum.solve(h, J, eps, gammas, exact=True, show=False)
        true_spec = true_spectrum.spectrum

    if ZERO:
        approx_spec -= approx_spec[:,0].reshape([-1,1])
        if FULL_RP:
            rp_spec -= rp_spec[:,0].reshape([-1,1])
        if EXACT_check:
            true_spec -= true_spec[:,0].reshape([-1,1])

    N = approx_spec.shape[1]
    if FULL_RP:
        N = min(N, rp_spec.shape[1])
    if EXACT_check:
        N = min(N, true_spec.shape[1])

    N -= 5
    # plotting
    plt.figure('Spectrum')

    plt.plot(s, approx_spec[:,:N], 'x', markersize=5)
    if FULL_RP:
        plt.plot(s, rp_spec[:,:N], 's')
    if EXACT_check:
        plt.plot(s, true_spec[:,:N], '-')

    plt.xlabel('Annealing Parameter (s)', fontsize=FS)
    plt.ylabel('$E$-$E_0$ (GHz)', fontsize=20)
    plt.ylim([0,15])

    if SAVE:
        plt.savefig(os.path.join(IMG_DIR, 'test_spectrum.eps'), bbox_inches='tight')

    plt.show()

    xl, xr = int(.3*nsteps), int(.47*nsteps)

    plt.figure('Zoom')

    plt.plot(s[xl:xr], approx_spec[xl:xr,:N], 'x', markersize=5)
    if FULL_RP:
        plt.plot(s[xl:xr], rp_spec[xl:xr,:N], 's')
    if EXACT_check:
        plt.plot(s[xl:xr], true_spec[xl:xr,:N], '-')

    plt.xlabel('Annealing Parameters (s)', fontsize=FS)
    plt.ylabel('$E$-$E_0$ (GHz)', fontsize=20)

    plt.xlim([s[xl],s[xr]])
    plt.ylim([1, 4.5])

    if SAVE:
        plt.savefig(os.path.join(IMG_DIR, 'test_zoom.eps'), bbox_inches='tight')

    plt.show()



    if False:
        # compare_spectra(approx_spectrum, rp_spectrum)
        if EXACT_check:
            compare_spectra(rp_spectrum, true_spectrum)
        elif FULL_RP:
            compare_spectra(approx_spectrum, true_spectrum)
    return
    locate_crossings(s, spec)


def from_coef(fname):
    ''' '''

    data = np.loadtxt(fname, dtype=float, delimiter=' ',
                      skiprows=1)

    rc = data[:,:2].astype(int)
    Z = data[:,2]

    J_ = defaultdict(dict)
    for (x,y), z in zip(rc, Z):
        J_[x][y] = z
        if x != y:
            J_[y][x] = z

    N = len(J_)
    h = np.zeros([N,], dtype=float)
    J = np.zeros([N, N], dtype=float)

    qbit_map = {k: i for i, k in enumerate(J_)}

    for x in J_:
        for y, z in J_[x].iteritems():
            if x==y:
                h[qbit_map[x]] = z
            else:
                J[qbit_map[x]][qbit_map[y]] = z

    yield h, J

def from_json(fname):
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

    yield h, J

def from_qca_file(fname, adj=None):
    ''' '''

    try:
        cells, spacing, zones, J, _ =  parse_qca_file(fname, one_zone=True)
    except:
        print('Failed to process QCA file: {0}'.format(fname))
        return

    # convert J to specified adjacency
    J = convert_adjacency(cells, spacing, J, adj=adj)

    # normalize J
    J /= np.max(np.abs(J))

    # group each type of cell: normal = NORMAL or OUTPUT
    drivers, fixeds, normals, outputs = [], [], [], []
    for i,c in enumerate(cells):
        if c['cf'] == CELL_FUNCTIONS['QCAD_CELL_INPUT']:
            drivers.append(i)
        elif c['cf'] == CELL_FUNCTIONS['QCAD_CELL_FIXED']:
            fixeds.append(i)
        else:
            if c['cf'] == CELL_FUNCTIONS['QCAD_CELL_OUTPUT']:
                outputs.append(i)
            normals.append(i)

    # map from output cell labels to indices in normals list

    J_d = J[drivers, :][:, normals]     # matrix for mapping drivers to h
    J_f = J[fixeds, :][:, normals]      # matrix for mapping fixed cells to h
    J_n = J[normals, :][:, normals]     # J internal to set of normal cells

    P_f = [(cells[i]['pol'] if 'pol' in cells[i] else 0.) for i in fixeds]  # polarization of fixed cells

    h0 = np.dot(P_f, J_f).reshape([-1,])    # h contribution from fixed cells

    # for each polarization, solve for gamma=0 and extimate remaining spectrum
    for pol in gen_pols(len(drivers)):
        h = h0 + np.dot(pol, J_d)
        yield h, J_n


def file_handler(fname, nsteps=100):
    '''Estimate spectrum for a file specified problem'''

    base_name = os.path.basename(fname)
    print(base_name)
    if re.match('.+\.txt', base_name) or re.match('.+\.coef', base_name):
        problems = from_coef(fname)
    elif re.match('.*\.json', base_name):
        problems = from_json(fname)
    elif re.match('.*\.qca', base_name):
        problems = from_qca_file(fname, adj=ADJ)
    else:
        problems = from_qca_file(fname, adj=ADJ)

    for h, J in problems:
        if False:
            find_true_spectrum(h, J, emin=0., nsteps=nsteps)
        else:
            estimate_spectrum(h, J, emin=0., nsteps=nsteps)

if __name__ == '__main__':

    try:
        fname = sys.argv[1]
    except:
        print('No input file...')
        sys.exit()

    try:
        nsteps = int(sys.argv[2])
    except:
        nsteps = 100

    file_handler(fname, nsteps=nsteps)
