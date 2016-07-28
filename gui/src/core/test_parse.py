from parse_qca import parse_qca_file

import sys
from pprint import pprint

def main(fname):

    try:
        cells, spacing, J = parse_qca_file(fname)
    except:
        print('Failed to load QCA file...')
        return

    pprint(cells)

if __name__ == '__main__':

    try:
        fname = sys.argv[1]
    except:
        print('No filename given...')
        sys.exit()

    main(fname)
