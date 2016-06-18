#!/usr/bin/env python

from parse_qca import parse_qca_file
from auxil import CELL_FUNCTIONS, gen_pols, convert_adjacency
from solvers.rp_solve import rp_solve, build_comp_H, compute_rp_tree
from solvers.sparse import solve_sparse
from scipy.signal import argrelmin

import numpy as np
import matplotlib.pyplot as plt
import sys, os, re, json

from collections import defaultdict
from time import time

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


ADJ = 'adj'
CACHE = './solvers/cache/rp_cache/0/'

ADAPTIVE_V0=False


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
    
    
    
def estimate_spectrum(h, J, gmax=1., emax=1., nsteps=100):
    ''' '''
    gmin = 1e-5
    gammas = np.linspace(gmax, gmin, nsteps)    # tranverse field scales
    eps = np.linspace(0, emax, nsteps)          # ising problem scales
    
    s = np.linspace(0, 1, nsteps)   # annealing schedule parameter
    
    # show recursive tree
    if False:
        tree = compute_rp_tree(J)
        show_tree(J, tree)

    # find initial modes
    print('Finding initial modes...'),
    t = time()
    e_vals, e_vecs, modes = rp_solve(h, J, gam=.5, verbose=True, 
                                     cache_dir=CACHE)
    print('{0:.2f} sec'.format(time()-t))
    
    
    # pre-compute sparse Hamiltonian components
    print('Pre-computing sparse Hamiltonian components...'),
    t = time()
    Hx = build_comp_H([h], [J], [], 1., [modes])
    diag = Hx.diagonal()
    Hx.setdiag([0]*diag.size)
    print('{0:.2f} sec'.format(time()-t))

#    rp_spectrum = []
    spectrum = []
    
    print('number of modes: {0}'.format(len(modes)))
    print('Estimating spectrum sweep...')
    
    v0 = None
    for i, (gamma, ep) in enumerate(zip(gammas, eps)):
        sys.stdout.write('\r{0:.2f}%'.format(i*100./nsteps))
        sys.stdout.flush()
        Hs = Hx*gamma
        Hs.setdiag(ep*diag)
        e_vals, e_vecs = solve_sparse(Hs, more=True, v0=v0)
        if ADAPTIVE_V0:
            v0 = e_vecs[:,0]
        spectrum.append(e_vals)
#        e_vals, e_vecs, m = rp_solve(h, J, gam=gamma)
#        rp_spectrum.append(e_vals)
    
    N = min(len(x) for x in spectrum)
#    N = min(N, min(len(x) for x in rp_spectrum))
    spectrum = [spec[:N] for spec in spectrum]
#    rp_spectrum = [spec[:N] for spec in rp_spectrum]
    
    spectrum = np.array(spectrum)
#    rp_spectrum = np.array(rp_spectrum)

    plt.figure('Spectrum')
    plt.plot(s, spectrum, 'x')
#    plt.plot(gammas, rp_spectrum, '-')

    plt.show(block=False)
    
    locate_crossings(s, spectrum)
    

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
#    output_map = {i: n for n, i in enumerate(normals) if i in outputs}
    
    J_d = J[drivers, :][:, normals]     # matrix for mapping drivers to h
    J_f = J[fixeds, :][:, normals]      # matrix for mapping fixed cells to h
    J_n = J[normals, :][:, normals]     # J internal to set of normal cells
    
    P_f = [(cells[i]['pol'] if 'pol' in cells[i] else 0.) for i in fixeds]  # polarization of fixed cells
    
    h0 = np.dot(P_f, J_f).reshape([-1,])    # h contribution from fixed cells
    
    # for each polarization, solve for gamma=0 and extimate remaining spectrum
    for pol in gen_pols(len(drivers)):
        h = h0 + np.dot(pol, J_d)
        yield h, J_n


def file_handler(fname):
    '''Estimate spectrum for a file specified problem'''
    
    base_name = os.path.basename(fname)
    if re.match('coefs[0-9]+\.txt', base_name):
        problems = from_coef(fname)
    elif re.match('.*\.json', base_name):
        problems = from_json(fname)
    elif re.match('.*\.qca', base_name):
        problems = from_qca_file(fname, adj=ADJ)
    else:
        problems = from_qca_file(fname, adj=ADJ)
    
    for h, J in problems:
        estimate_spectrum(h, J)
    
if __name__ == '__main__':
    
    try:
        fname = sys.argv[1]
    except:
        print('No input file...')
        sys.exit()
    
    file_handler(fname)