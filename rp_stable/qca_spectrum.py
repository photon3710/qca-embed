#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from solvers.spectrum import Spectrum
from parse_qca import parse_qca_file
from auxil import convert_adjacency, qca_to_coef, normalized_coefs

import types
import sys


def linear_clock(**params):
    '''Linear clock from t0 to tf in N steps'''

    s = np.linspace(params['t0'], params['tf'], params['N'])

    eps = s
    gamma = 1.-s

    return s, eps, gamma

def clock_schedule(scheme=None, **params):

    # assert necessary parameters
    def assert_par(nm, val):
        if nm not in params:
            print('schedule parameters: no {0} given... setting to {1}'.format(nm, val))
            params[nm]=val

    assert_par('t0', 0.)
    assert_par('tf', 1.)
    assert_par('N', 100)

    # select clock schedule and generate
    if scheme is None or scheme == 'linear':
        return linear_clock(**params)


def find_spectrum(fname, adj=None):
    ''' '''

    cells, spacing, zones, J, feedback = parse_qca_file(fname, one_zone=True)
    h, J = qca_to_coef(cells, spacing, J, adj=adj)
    h, J = normalized_coefs(h,J)

    # get clocking schedule
    s, eps, gamma = clock_schedule(scheme='linear')

    spectrum = Spectrum()

    spectrum.solve(h, J, eps, gamma, show=True, exact=False)


if __name__ == '__main__':

    # QCADesigner file
    try:
        fn = sys.argv[1]
    except:
        print('No input file given...')
        sys.exit()

    # adjacency type
    adjs = ['lim', 'full', None]
    if len(sys.argv)>2 and isinstance(sys.argv[2], types.IntType):
        n = sys.argv[2]
    else:
        n = 0
    n = max(min(n, 2), 0)

    # run
    find_spectrum(fn, adj=adjs[n])
