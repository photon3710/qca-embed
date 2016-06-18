import numpy as np

from parse_qca import parse_qca_file
from auxil import CELL_FUNCTIONS, gen_pols, convert_adjacency
from solvers.core import state_to_pol
from solvers.sparse import solve

import sys

def exact_solve(fname, gammas = [0.], k=-1, adj=None):
    '''Exactly solve the first k eigenstates for a QCA circuit for all possible
    input configurations and all specified transverse fields. Assumes all cells
    have the same transverse field.'''
    
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
            e_vals, e_vecs = solve(h, J_n, gamma=gamma, minimal=True, exact=False)
            # e_vecs[:,i] is the i^th eigenstate
            pols = state_to_pol(e_vecs)
            # pols[:,i] gives the polarizations of all cells for the i^th e-vec
            print('GS: {0:.4f}'.format(e_vals[0]))
            print('pols: {0}'.format(pols[:,0]))
#            for o, i in output_map.iteritems():
#                print('{0}: {1}'.format(o, pols[:,0]))
    
if __name__ == '__main__':
    try:
        fname = sys.argv[1]
    except:
        print('No QCA file given...')
        sys.exit()
        
    exact_solve(fname, gammas = [0.,], adj=None)