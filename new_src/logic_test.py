#!/usr/bin/env python

import sys
import os
import json

import numpy as np
from pprint import pprint

from auxil import logical_chains, to_logical

def proc_json(fname):
    '''Recover parameters from the json file '''
    
    try:
        fp = open(fname, 'r')
    except:
        print('Failed to open file: {0}'.format(os.path.basename(fname)))
        return
        
    data = json.load(fp)
    fp.close()
    
    qbits = data['qbits']
    qbit_map = {str(qb): n for n, qb in enumerate(qbits)}
    
    N = len(qbits)  # number of qubits
    h = np.zeros([N,], dtype=float)
    J = np.zeros([N,N], dtype=float)
    
    for qb, val in data['h'].iteritems():
        h[qbit_map[qb]] = val

    for qb1 in data['J']:
        for qb2, val in data['J'][qb1].iteritems():
            J[qbit_map[qb1], qbit_map[qb2]] = val
            J[qbit_map[qb2], qbit_map[qb1]] = val
            
    Es = data['energies']
    spins = np.array(data['spins']).T
    occ = data['occ']
    
    return h, J, Es, spins, occ

def main(fname):
    ''' '''
    
    h, J, Es, spins, occ = proc_json(fname)

    chains = logical_chains(h, J)
    logical = to_logical(spins, chains)

    
    
    
    


if __name__ == '__main__':
    
    try:
        fname = sys.argv[1]
    except:
        print('No json file given')
        sys.exit()
        
    main(fname)