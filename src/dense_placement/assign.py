#!/usr/bin/python

# -----------------------------------------------------------------------------
# Name: assign.py
# Purpose:  qubit and coupler parameter assignment
# Author:   Jacob Retallick
# Created:  24.06.2015
# Last modified: 24.06.2015
# -----------------------------------------------------------------------------

import numpy as np
import networkx as nx
import itertools
import re

# ASSIGNMENT PARAMETERS
J_INNER = -1    # J parameter for couplers within subgraphs


strategies = ['maximal',    # Use as many couplers as possible
              'minimal'     # Use as few couplers as possible
              ]

strategy = 'maximal'
assert strategy.lower() in strategies, 'Invalid edge selection strategy'


def sort_dict(d):
    '''Sort the keys of a dict by d[key]'''
    return zip(*sorted([(d[k], k) for k in d]))[1]


def pinch(s, s1, s2):
    return s.partition(s1)[2].rpartition(s2)[0]


# Edge selection strategies


def maximal_couplers(subgraphs, edges):
    '''Use as many edges as possible between and within subgraphs'''
    return subgraphs, edges


def minimal_couplers(subgraphs, edges):
    '''Use the fewest possible number of couplers between and within
    subgraphs'''

    N = len(subgraphs)

    # map each subgraph to its minimum spanning tree
    subgraphs = [nx.minimum_spanning_tree(subgraph) for subgraph in subgraphs]

    # for each tree, find a root node and store the shortest path to each
    # node as a cost metric.
    costs = {}
    for tree in subgraphs:
        # identify the root
        path_lengths = nx.shortest_path_length(tree)
        root_weights = {k: sum(path_lengths[k].values()) for k in path_lengths}
        root = sort_dict(root_weights)[0]
        # assign path lengths as node costs
        for node in path_lengths[root]:
            costs[node] = path_lengths[root][node]

    # for each pair of subgraphs, keep the inter-subgraph edge with the
    # minimum total cost of its end nodes
    for i in xrange(N-1):
        for j in xrange(i+1, N):
            edge_costs = {e: costs[e[0]]+costs[e[1]] for e in edges[(i, j)]}
            edges[(i, j)] = sort_dict(edge_costs)[0]

    return subgraphs, edges


#


def write_to_file(hq, Jq, fn):
    '''Write the parameters to file.

    inputs: hq  : dictionary of h parameters for used qubits
            Jq  : dictionary of J parameters for used couplers
            fn  : file path
    '''

    try:
        fp = open(fn, 'w')
    except IOError:
        print('Invalid file path... {0}'.format(fn))
        return None

    fp.write('<parameters>\n')

    # write the h parameters
    fp.write('<h>\n')
    for qbit in hq:
        fp.write('\t{0}:\t{1}\n'.format(qbit, hq[qbit]))
    fp.write('</h>\n\n')

    # write the J parameters
    fp.write('<J>\n')
    for coupler in Jq:
        fp.write('\t{0}:\t{1}\n'.format(coupler, Jq[coupler]))
    fp.write('</J>\n\n')

    fp.write('</parameters>')


def read_from_file(fn):
    '''Read the h and J parameters of an embedding from file'''

    try:
        fp = open(fn, 'w')
    except IOError:
        print('Invalid file path... {0}'.format(fn))
        return None

    # reading flags
    reading_h = False
    reading_J = False

    # regex
    re_start = re.compile('^<\w*>$')
    re_end = re.compile('^</\w*>$')

    hq = {}
    Jq = {}

    for line in fp:
        if '#' in line or len(line) < 3:
            continue
        if re_start.match(line.strip()):
            key = pinch(line, '<', '>').strip()
            if key == 'h':
                reading_h = True
            elif key == 'J':
                reading_J = True
        elif re_end.match(line.strip()):
            key = pinch(line, '</', '>').strip()
            if key == 'h':
                reading_h = False
            elif key == 'J':
                reading_J = False
        elif reading_h:
            qbit, h = line.strip().split(':')
            qbit = int(qbit)
            hq[qbit] = float(h)
        elif reading_J:
            coupler, J = line.strip().split(':')
            coupler = tuple(map(int, pinch(coupler, '(', ')').split(',')))
            Jq[coupler] = float(J)

    return hq, Jq


def partition_graph(G, parts):
    '''Partition graph G into a set of disjoint subgraphs given by a list of
    lists of node labels for each subgraph

    inputs: G       : Graph object to be partitioned
            parts   : list of lists of subgraph nodes. Must have same labels as
                     in G
    '''

    N = len(parts)  # number of partitions

    # get partition subgraphs

    subgraphs = {}
    for i, part in enumerate(parts):
        subgraph = G.subgraph(part)
        if len(subgraph) < len(part):
            conflicts = [n for n in part if not n in subgraph.nodes()]
            raise KeyError('Invalid indices given: {0}'.format(conflicts))
        subgraphs[i] = subgraph

    # get list of edges between each subgraph
    edges = {}
    for i1 in xrange(N-1):
        for i2 in xrange(i1+1, N):
            edges[(i1, i2)] = []
            for u, v in list(itertools.product(parts[i1], parts[i2])):
                edge = sorted([u, v])
                if G.has_edge(*edge):
                    edges[(i1, i2)].append(edge)
            if len(edges[(i1, i2)]) == 0:
                raise KeyError('No edges found between partitions {0} \
                                and {1}'.format(i1, i2))

    return subgraphs, edges


def convert_to_parameters(h, J, subgraphs, edges):
    '''Construct parameter dictionaries from the problem h and J coefficients
    and the determined subgraphs and edges'''

    assert len(h) == len(subgraphs),\
        'Mismatch between problem nodes and subgraphs'

    N = len(h)

    hq = {}     # h parameter for each qubit used: keys are integers
    Jq = {}     # J parameter for each coupler: key format (u, v) with u < v

    # handle internal subgraph parameters
    for i, subgraph in enumerate(subgraphs):
        for node in subgraph.nodes():
            hq[node] = h[i]*1./subgraph.number_of_nodes()
        for edge in subgraph.edges():
            Jq[edge] = J_INNER

    # handle inter-subgraph parameters
    for i in xrange(N-1):
        for j in xrange(i+1, N):
            for edge in edges[(i, j)]:
                Jq[edge] = J[i, j]*1./len(edges[(i, j)])

    return hq, Jq


def assign_parameters(h, J, qbits, chimera, flip_J=False):
    '''Determine the h and J coefficients for a given embedding problem given
    a list of qubits for each problem node and an adjacency list for the
    target chimera structure (with qbit labels as verticed)

    inputs: h       : list of on-site terms for each problem node
            J       : list of coupling terms between each problem node
            qbits   : list of qbits for each problem node
            chimera : adjacency list structure for the target chimera graph
            flip_J   : flag for flipping the sign of J
    '''

    # check that h and J are normalised
    max_hj = max(np.max(np.abs(h)), np.max(np.abs(J)))
    if max_hj == 0:
        print('Invalid problem statement. All zero parameters')
        return None
    if max_hj != 1:
        print('Normalising h and J by factor {0}'.format(max_hj))
        h = (h/max_hj).tolist()
        J = J/max_hj

    # flip J signs if flagged
    if flip_J:
        print('Flipping signs of J coefficients')
        J = -J

    # build chimera graph
    G_chimera = nx.Graph(chimera)

    # get subgraphs and edge lists for problem node qbit lists
    try:
        subgraphs, edges = partition_graph(G_chimera, qbits)
    except KeyError as e:
        print('Qbit label error during Chimera graph partition...')
        print(e.message)
        return None

    # remove unwanted edges
    if strategy.lower() == 'maximal':
        subgraphs, edges = maximal_couplers(subgraphs, edges)
    elif strategy.lower() == 'minimal':
        subgraphs, edges = minimal_couplers(subgraphs, edges)
    else:
        print('No valid edge selection strategy given...')
        return None

    # convert subgraphs and edges to parameter dictionaries
    hq, Jq = convert_to_parameters(h, J, subgraphs, edges)

    # return parameter
    return hq, Jq
