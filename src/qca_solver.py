import numpy as np
import scipy.sparse as sp

from parse_qca import parse_qca_file
from auxil import CELL_FUNCTIONS
from solve import solve

def polarizations(n):
    '''compute list of all possible polarizations for n cells'''
    if n <= 0:
        return []
    return [tuple(2*int(x)-1 for x in format(i, '#0{0}b'.format(n+2))[2:])
            for i in range(pow(2, n))]

def exact_solve(fname, gammas=[0.], k=-1):
    '''Exactly solve the first k eigenstates for a QCA circuit for all possible
    input configurations and all specified transverse fields. Assumes all cells
    have the same transverse field.'''
    
    # process the QCADesigner file
    try:
        cells, spacing, zones, J, _ = parse_qca_file(fname, one_zone=True)
    except:
        print('Failed to process QCA file: {0}'.format(fname))
        return None
    
    # isolate each type of cell
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
    
    # map  from output label to index in list of normals
    output_map = {i: n for n, i in enumerate(normals) if i in outputs}
    
    J_d = J[drivers, :][:, normals]     # matrix for mapping drivers to h
    J_f = J[fixeds, :][:, normals]      # matrix for mapping fixed cells to h
    J_n = J[normals, :][:, normals]     # J internal to set of normal cells
    
    P_f = [(cells[i]['pol'] if 'pol' in cells[i] else 0.) for i in fixeds]
    
    h0 = np.dot(P_f, J_f).reshape([-1,])

    # for each polarization, solve for all gammas
    for pol in polarizations:
        h = h0 + np.dot(pol, J_d)
        sols = {}
        for gamma in gammas:
            pass
            
        
    
                