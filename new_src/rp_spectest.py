#!/usr/bin/env python

import numpy as np
import sys, os

import matplotlib.pyplot as plt
from time import time

from solvers.rp_solve import rp_solve
from solvers.sparse import solve

from parse_qca import parse_qca_file
from auxil import CELL_FUNCTIONS, gen_pols, convert_adjacency

FS = 14
CACHE = './solvers/cache/temp'
STEPS = 100

def compare_spectrum(fname, gmin=0.01, gmax=1., adj='lim'):
    ''' '''
    
    try:
        cells, spacing, zones, J, _ = parse_qca_file(fname, one_zone=True)
    except:
        print('Failed ot process QCA file: {0}'.format(fname))
        return None
        
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
    output_map = {i: n for n, i in enumerate(normals) if i in outputs}
    
    J_d = J[drivers, :][:, normals]     # matrix for mapping drivers to h
    J_f = J[fixeds, :][:, normals]      # matrix for mapping fixed cells to h
    J_n = J[normals, :][:, normals]     # J internal to set of normal cells
    
    P_f = [(cells[i]['pol'] if 'pol' in cells[i] else 0.) for i in fixeds]  # polarization of fixed cells
    
    h0 = np.dot(P_f, J_f).reshape([-1,])    # h contribution from fixed cells
    
    gammas = np.linspace(gmin, gmax, STEPS)
    
    for pol in gen_pols(len(drivers)):
        h = h0 + np.dot(pol, J_d)
        SP_E, RP_E = [], []
        for gamma in gammas:
            print(gamma)
            rp_vals, rp_vecs, modes = rp_solve(h, J_n, gam=gamma, 
                                               cache_dir=CACHE)
            sp_vals, sp_vecs = solve(h, J_n, gamma=gamma)
            SP_E.append(sp_vals)
            RP_E.append(rp_vals)
        LSP = min(len(x) for x in SP_E)
        L = min(LSP, min(len(x) for x in RP_E))
        SP_E = np.array([x[:L] for x in SP_E])
        RP_E = np.array([x[:L] for x in RP_E])
        
        plt.plot(gammas, SP_E, 'g', linewidth=2)
        plt.plot(gammas, RP_E, 'bx', markersize=8, markeredgewidth=2)
        plt.xlabel('Gamma', fontsize=FS)
        plt.ylabel('Energy', fontsize=FS)
        plt.title('Circuit Spectrum', fontsize=FS)
        plt.legend(['Exact', 'RP-Solver'], fontsize=FS)
        plt.show()
    
if __name__ == '__main__':
    try:
        fname = sys.argv[1]
    except:
        print('No QCA file given...')
        sys.exit()
    compare_spectrum(fname)