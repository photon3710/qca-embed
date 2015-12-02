#!/usr/bin/env python

#---------------------------------------------------------
# Name: chimera.py
# Purpose: Functions for handling the chimera graph
# Author:	Jacob Retallick
# Created: 11.30.2015
# Last Modified: 11.30.2015
#---------------------------------------------------------

import numpy as np

# constants
L = 4   # number of qubits per half tile


def linear_to_tuple(ind, M, N, L=4, index0=False):
    '''Convert the linear index of a qubit in an (N, M, L) processor to 
    tuple format'''
    
    qpr = 2*N*L     # qbits per row
    qpt = 2*L       # qbits per tile

    if not index0:
        ind -= 1

    row, rem = divmod(ind, qpr)
    col, rem = divmod(rem, qpt)
    horiz, ind = divmod(rem, L)

    return (row, col, horiz, ind)

    
def tuple_to_linear(tup, M, N, L=4, index0=False):
    '''Convert a tuple format index of a qubit in an (N, M, L) processor
    to linear format'''
    
    qpr = 2*N*L     # qbits per row
    qpt = 2*L       # qbits per tile

    return (0 if index0 else 1) + qpr*tup[0]+qpt*tup[1]+L*tup[2]+tup[3]


def load_chimera_file(filename):
    '''Load a chimera graph from an edge specification file'''
    
    try:
        fp = open(filename, 'r')
    except:
        print('Failed to open file: {0}'.format(filename))
        raise IOError
        
    # get number of qubits and number of connectors
    num_qbits, num_conns = [int(x) for x in fp.readline().split()]
    
    adj = {i: [] for i in xrange(1, num_qbits+1)}

    for line in fp:
        a, b = [int(x) for x in line.strip().split()]
        adj[a].append(b)
        adj[b].append(a)
    
    # processor size
    M = int(np.sqrt(num_qbits/(2*L)))
    N = M
    
    return M, N, adj
    
    