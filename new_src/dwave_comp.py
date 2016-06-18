#!/usr/bin env python

import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

import json
import sys

from solvers.rp_solve import rp_solve, compute_rp_tree

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
        
def compare(fname):
    '''Compare stored solution from D-Wave annealer to computed solutions'''
    
    try:
        fp = open(fname, 'r')
    except:
        print('Failed to open file: {0}'.format(fname))
        return
        
    data = json.load(fp)
    
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
    
    tree = compute_rp_tree(J)
    show_tree(J, tree)    

    G = nx.Graph(J)
    pos = graphviz_layout(G)
    nx.draw(G, pos=pos, with_labels=True)
    plt.show()
    
    e_vals, e_vecs, modes = rp_solve(h, J, gam=0.01)
    
    print(e_vals)
    print(data['energies'])


if __name__ == '__main__':
    try:
        fname = sys.argv[1]
    except:
        print('No file given...')
        sys.exit()

    compare(fname)