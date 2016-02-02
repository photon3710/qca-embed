#!/usr/bin/python

from gui.qca_widget import test_app
from parse_qca import parse_qca_file
from numpy.random import random

import sys

if __name__ == '__main__':
    try:
        fn = sys.argv[1]
    except:
        print('no file entered...')
        sys.exit()

    cells, spacing, zones, J = parse_qca_file(fn)
    N = len(cells)
    pols = [2*random()-1 for _ in xrange(N)]
    parts = [int(random()*6)-1 for _ in xrange(N)]
    test_app(cells, spacing, pols, parts, style='pols')
