from parse_qca import parse_qca_file
from auxil import CELL_FUNCTIONS

import sys, os
from pprint import pprint

def count_cells(cells):
    '''count the non-driver cells in a list of cells'''

    ncells, ninputs = 0, 0

    cell_keys = [CELL_FUNCTIONS['QCAD_CELL_NORMAL'],
                 CELL_FUNCTIONS['QCAD_CELL_OUTPUT']]

    input_keys = [CELL_FUNCTIONS['QCAD_CELL_INPUT']]

    for cell in cells:
        if cell['cf'] in cell_keys:
            ncells += 1
        elif cell['cf'] in input_keys:
            ninputs += 1

    return ncells, ninputs

def main(fname):

    try:
        cells, spacing, J = parse_qca_file(fname)
    except:
        print('Failed to load QCA file...')
        return

    name = os.path.basename(fname).ljust(35)
    ncells, ninputs = count_cells(cells)
    print('{0}: {1} cells,\t{2} inputs'.format(name, ncells, ninputs))

def main_dir(dirname):

    for fname in os.listdir(dirname):
        fn = os.path.join(dirname, fname)
        main(fn)

if __name__ == '__main__':

    try:
        fname = sys.argv[1]
    except:
        print('No filename given...')
        sys.exit()

    main_dir(os.path.dirname(fname))
