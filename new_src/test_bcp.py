from parse_qca import parse_qca_file
from auxil import convert_adjacency
from solvers.bcp import chlebikova
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

import sys

ADJ = 'full'

def main(fname):
    
    try:
        cells, spacing, zones, J, _ = parse_qca_file(fname, one_zone=True)
    except:
        print('Failed to process QCA file: {0}'.format(fname))
        return None
    
    # convert J to specified adjacency
    J = 1.*(convert_adjacency(cells, spacing, J, adj=ADJ) != 0)
    
    G = nx.Graph(J)
    for k in G:
        G.node[k]['w'] = 1./len(G[k])**2
        
    pos = graphviz_layout(G)
    nx.draw(G, pos=pos, with_labels=True)
    plt.show()
    
    chlebikova(G)
    
    
if __name__ == '__main__':
    
    try:
        fname = sys.argv[1]
    except:
        print('No filename given...')
        sys.exit()
    
    main(fname)