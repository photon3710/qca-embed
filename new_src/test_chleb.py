import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from pprint import pprint
from itertools import chain

from parse_qca import parse_qca_file
from auxil import convert_adjacency
from solvers.bcp import blockbalance

import sys, os

COEF_ROOT = os.path.join(os.path.pardir, 'dat', 'coef_files')
SAVE_COEF = True


def to_coef(fname, J):
    ''' Save J data to a coef file '''
    
    fname = os.path.join(COEF_ROOT, fname)
    
    try:
        fp = open(fname, 'w')
    except IOError:
        print('Failed to open file: {0}'.format(os.path.basename(fname)))
        
    N = J.shape[0]
    fp.write('{0}\n'.format(N))
    
    for i in range(N-1):
        for j in range(i+1,N):
            if J[i][j] != 0:
                fp.write("{0} {1} {2:.3f}\n".format(i, j, J[i][j]))
    
    fp.close()
    
def load_qca(fname):
    
    cells, spacing, zones, J, feedback = parse_qca_file(fname, one_zone=True)
    
    J = convert_adjacency(cells, spacing, J, adj='full')
    J = -J/np.max(np.abs(J))
    
    if SAVE_COEF:
        fname = os.path.splitext(os.path.basename(fname))[0]
        fname += os.path.extsep + 'coef'
        to_coef(fname, J)

    return J
    
    
def load_tri(fname):
    
    try:
        fp = open(fname, 'r')
    except IOError:
        print('Failed to open file: {0}'.format(fname))
    
    fp.readline()

    _J = defaultdict(dict)
    for line in fp:
        i, j, v = line.strip().split()
        i, j = map(int, [i,j])
        v = float(v)
        _J[i][j] = _J[j][i] = v
    fp.close()
    
    qbit_map = {k: x for x, k in enumerate(sorted(_J))}
    
    J = defaultdict(dict)
    for i, adj in _J.items():
        for j, v in adj.items():
            J[qbit_map[i]][qbit_map[j]] = v

    return dict(J)


# ARTICULATION POINT ALGORITHM FROM NETWORKX     
    
def biconnected_dfs(G, components=True):
    # depth-first search algorithm to generate articulation points
    # and biconnected components
    visited = set()
    for start in G:
        if start in visited:
            continue
        discovery = {start: 0}  # time of first discovery of node during search
        low = {start: 0}
        root_children = 0
        visited.add(start)
        edge_stack = []
        stack = [(start, start, iter(G[start]))]
        while stack:
            grandparent, parent, children = stack[-1]
            try:
                child = next(children)
                if grandparent == child:
                    continue
                if child in visited:
                    if discovery[child] <= discovery[parent]:  # back edge
                        low[parent] = min(low[parent], discovery[child])
                        if components:
                            edge_stack.append((parent, child))
                else:
                    low[child] = discovery[child] = len(discovery)
                    visited.add(child)
                    stack.append((parent, child, iter(G[child])))
                    if components:
                        edge_stack.append((parent, child))
            except StopIteration:
                stack.pop()
                if len(stack) > 1:
                    if low[parent] >= discovery[grandparent]:
                        if components:
                            ind = edge_stack.index((grandparent, parent))
                            yield edge_stack[ind:]
                            edge_stack = edge_stack[:ind]
                        else:
                            yield grandparent
                    low[grandparent] = min(low[parent], low[grandparent])
                elif stack:  # length 1 so grandparent is root
                    root_children += 1
                    if components:
                        ind = edge_stack.index((grandparent, parent))
                        yield edge_stack[ind:]
        if not components:
            # root node is articulation point if it has more than 1 child
            if root_children > 1:
                yield start

def main(fname):
    
    if os.path.splitext(fname)[-1][1::].lower() == 'qca':
        J = load_qca(fname)
    else:
        J = load_tri(fname)
    
    G = nx.Graph(J)
    
    for k in G:
        G.node[k]['w'] = 1.
    
    if False:
        for i in G:
            print i
        for x in biconnected_dfs(G, components=False):
            print(x)
    else:
        V1, V2, B = blockbalance(G)
        
        print(V1)
        print(V2)
    

if __name__ == '__main__':
    
    try:
        fname = sys.argv[1]
    except:
        print('No file given...')
        sys.exit()
    
    main(fname)