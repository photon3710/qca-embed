#!/usr/bin/env python

import networkx as nx
from collections import defaultdict, Iterable
from itertools import product

from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from heapq import heappop, heappush

from pprint import pprint

def gen_dfs(G, node, visited=None):
    '''generate dfs order for a given graph G starting at the specified node
    and with optional dict of visited flags. G[v] should be an iterable of
    all the neighbours of v'''
    
    if visited is None:
        visited = {k: False for k in G}
    
    assert node in G, 'start node not in G'
    stack = [node]
    
    while stack:
        v = stack.pop()
        if visited[v]:
            continue
        visited[v] = True
        for x in G[v]:
            if not visited[x]:
                stack.append(x)
        yield v

def blockbalance(G):
    '''Find heuristic BCP2(G) where G is a 2-connected networkx Graph '''

    V1, V2 = set(), set(G.nodes())
    w_diff = sum(G.node[k]['w'] for k in G)
    
    # flags
    visited = {v: False for v in G}
    cut_vertex = {v: False for v in G}
    ap = []     # current list of articulation points
    
    # start with highest weight node
    start = max(V2, key=lambda x: G.node[x]['w'])
    
    # maintain lowest weight candidate node in a priority queue.
    queue = [(G.node[start]['w'], start)]
    
    while True:
        # pop next node to add to V1
        w, v = heappop(queue)
        if visited[v] or cut_vertex[v]:
            continue
        # update balance metric and check break condition
        w_diff -= 2*w
        if V1 and w_diff < 0:
            break
        # if passed, move v to V1
        visited[v] = True
        V1.add(v)
        V2.remove(v)
        # update articulation points
        for x in ap:
            cut_vertex[x]=False
        ap = list(nx.articulation_points(G.subgraph(V2)))
        for x in ap:
            cut_vertex[x]=True
        for node in G[v]:
            if visited[node] or cut_vertex[node]:
                continue
            heappush(queue, (G.node[node]['w'], node))
            
    # compute balance measure
    W1 = sum(G.node[x]['w'] for x in V1)
    W2 = sum(G.node[x]['w'] for x in V2)
    B = W1*W2

    return V1, V2, B
    
def block_handler(G, T, h):
    '''Compute block effective node weights and keep track of nodes in each 
    branch'''
    
    assert h in T, 'block-T mismatch'
    
    visited = {k: False for k in T}
    visited[h] = True
    
    # tree traversal from h in T
    branches = T[h] # cut_vertices out of the block
    
    # propagate along each branch
    nodes = defaultdict(set)
    for branch in branches:
        order = list(gen_dfs(T, branch, visited=visited))
        for x in order:
            if isinstance(x, Iterable):
                nodes[branch].update(x)

    # create block subgraph    
    Gh = G.subgraph(h)
    
    # update branch weights
    old_w = {}
    for branch in branches:
        old_w[branch] = Gh.node[branch]['w']
        Gh.node[branch]['w'] = sum(G.node[x]['w'] for x in nodes[branch])
    
    # bisect Gh
    V1, V2, B =  blockbalance(Gh)
    
    # reset branch weights
    for branch in branches:
        Gh.node[branch]['w'] = old_w[branch]

    return V1, V2, B, nodes

def chlebikova(G):
    '''Use Chlebikova's algorithm to estimate an optimal bisection of a 
    networkx graph G with the weight of node k as G.node[k]['w']'''

    # Assert graph format and structure
    assert isinstance(G, nx.Graph), 'G is not an undirected nx.Graph'
    assert nx.number_connected_components(G) == 1, 'G is not connected'
    
    # Handle default weighting
    for k in G:
        if 'w' not in G.node[k]:
            print('No weight given for node {0}, assigning w = 1'.format(k))
            G.node[k]['w'] = 1
            
    # list of articulation points
    A = set(nx.articulation_points(G))
    
    # set of all biconnected components
    H = list(nx.biconnected_components(G))
    H = [tuple(x) for x in H]
    
    # contruct block-articulation graph
    T = defaultdict(list)
    for a, h in product(A, H):
            if a in h:
                T[a].append(h)
                T[h].append(a)
    
    if not T:
        V1, V2, B = blockbalance(G)
    else:
        # try all blocks and keep most balanced
        B_max = 0.
        V1, V2, nodes = None, None, None
        for h in H:
            V1_, V2_, B, nodes_ = block_handler(G, T, h)
            if B > B_max:
                V1, V2, nodes, B_max = V1_, V2_, nodes_, B
        
        # determine which partition each branch belongs to
        parts = [[], []]
        for k in nodes:
            if k in V1:
                parts[0].append(k)
            else:
                parts[1].append(k)
        
        for k in parts[0]:
            V1.update(nodes[k])
        for k in parts[1]:
            V2.update(nodes[k])
        
    return V1, V2
    
    


