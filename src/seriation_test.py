#!/usr/bin/env python

from parse_qca import parse_qca_file

import sys


def main(fname):
    pass    


if __name__ == '__main__':
    try:
        fname = sys.argv[1]
    except KeyError:
        print('No QCA file given')
    
    if fname:
        main(fname)