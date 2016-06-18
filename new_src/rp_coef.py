#!/usr/bin/env python

import numpy as np
from solvers.rp_solve import rp_solve, compute_rp_tree
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from collections import defaultdict
import sys
from pprint import pprint

CACHE = './solvers/cache/temp'

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
    
def load_coef_file(fname):
    
    plt.close()

    try:
        fp = open(fname, 'r')
    except:
        print('Failed to open coef file')
        return None
    
    # burn number of qbits
    fp.readline()
    
    h_ = {}
    J_ = defaultdict(dict)
    
    qbits = set()
    for line in fp:
        i, j, v = line.strip().split()
        i, j, v = int(i), int(j), float(v)
        qbits.add(i)
        if i==j:
            h_[i] = v
        else:
            qbits.add(j)
            J_[i][j] = v
            J_[j][i] = v
    
    N = len(qbits)
    qbit_map = {qb: k for k, qb in enumerate(qbits)}

    h = np.zeros([N,], dtype=float)
    J = np.zeros([N, N], dtype=float)
    
    for qb, v in h_.iteritems():
        h[qbit_map[qb]] = v
    
    for q1 in J_:
        for q2, v in J_[q1].iteritems():
            J[qbit_map[q1], qbit_map[q2]] = v
            J[qbit_map[q2], qbit_map[q1]] = v
    
#    tree = compute_rp_tree(J, nparts=2)
#    show_tree(J, tree)
    
    return h, J

def solve_coef_file(fname):
    '''solve a coef file using the rp_solver'''
    
    h, J = load_coef_file(fname)

    e_vecs, e_vals, modes = rp_solve(h, J, gam=0.01, cache_dir=None, verbose=True)
    
    print(e_vecs)
    
if __name__ == '__main__':
    
    try:
        fname = sys.argv[1]
    except:
        print('Noe coef file given')
        sys.exit()
    
    solve_coef_file(fname)