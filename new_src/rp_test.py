#!/usr/bin.env python

import numpy as np
import sys

from parse_qca import parse_qca_file
from auxil import CELL_FUNCTIONS, gen_pols, convert_adjacency
from solvers.core import state_to_pol
from solvers.sparse import solve
from solvers.rp_solve import rp_solve
#from solvers.old_rp_solve import rp_solve

from time import time

def approx_solve(fname, gammas = [0.], k=-1, adj=None):
    ''' '''
    # process the QCADesigner file
    try:
        cells, spacing, zones, J, _ = parse_qca_file(fname, one_zone=True)
    except:
        print('Failed to process QCA file: {0}'.format(fname))
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
    
    print(fixeds)
    # map from output cell labels to indices in normals list
    output_map = {i: n for n, i in enumerate(normals) if i in outputs}
    
    J_d = J[drivers, :][:, normals]     # matrix for mapping drivers to h
    J_f = J[fixeds, :][:, normals]      # matrix for mapping fixed cells to h
    J_n = J[normals, :][:, normals]     # J internal to set of normal cells
    
    P_f = [(cells[i]['pol'] if 'pol' in cells[i] else 0.) for i in fixeds]  # polarization of fixed cells
    
    h0 = np.dot(P_f, J_f).reshape([-1,])    # h contribution from fixed cells
    
    # for each polarization, solve for all gammas
    for pol in gen_pols(len(drivers)):
        h = h0 + np.dot(pol, J_d)
        for gamma in gammas:
            t = time()
            e_vals, e_vecs, modes = rp_solve(h, J_n, gam=gamma, verbose=True)
            print('\n')
            print(e_vals[0:2], time()-t)
            if False:
                t = time()
                e_vals, e_vecs = solve(h, J_n, gamma=gamma, minimal=False, exact=False)
                print(e_vals[0:2], time()-t)

    
if __name__ == '__main__':
    
    try:
        fname = sys.argv[1]
    except:
        print('No file given...')
        sys.exit()
    gammas = [.1*n for n in range(1, 2)]
    approx_solve(fname, gammas=gammas, adj='full')