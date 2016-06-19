#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from solvers.spectrum import Spectrum
from collections import defaultdict
from pprint import pprint

import os, sys, json, re

OUT_FILE = os.path.join(os.getcwd(), 'data', 'xings.dat')
print(OUT_FILE)

def get_files(direc):
    '''get all the files nested under a directory'''

    regex = re.compile('.*\.json')

    fnames = []
    for root, dirs, files in os.walk(direc):
        for fn in files:
            if regex.match(fn):
                fname = os.path.join(root, fn)
                fnames.append(fname)

    return fnames

def load_coef_file(fname):
    ''' Process a tri-column coef file into h and J dicts'''

    try:
        fp = open(fname, 'r')
    except IOError:
        print('Failed to load file: {0}'.format(fname))
        raise IOError

    nqbits = int(fp.readline())
    print('Loading coef file with {0} qbits'.format(nqbits))

    h = defaultdict(float)
    J = defaultdict(dict)
    for line in fp:
        a, b, v = line.split()
        a, b, v = int(a), int(b), float(v)
        if a==b:
            h[a] = v
        else:
            a, b = sorted([a, b])
            J[a][b] = v
    fp.close()

    return h, J

def load_json_file(fname):
    ''' Pull data from a json formatted D-Wave result file '''

    try:
        fp = open(fname, 'r')
    except IOError:
        print('Failed to load file: {0}'.format(fname))
        raise IOError

    data = json.load(fp)
    fp.close()

    qbits = data['qbits']
    energies = data['energies']
    spins = data['spins']
    occ = data['occ']

    return qbits, energies, spins, occ

def append_data(params, rt, pfact):
    ''' Append the new xing data to the cache '''

    try:
        fp = open(OUT_FILE, 'a')
    except:
        print('Failed to open file: {0}...'.format(OUT_FILE))
        return

    for p, d0, w0 in filter(None, params):
        fp.write('{0} {1} {2} {3} {4}\n'.format(rt, pfact, p, d0, w0))
    fp.close()

def main(fname, nsteps=100):
    ''' '''

    print('\n\nRunning problem: {0}...'.format(os.path.splitext(fname)[0]))

    coef_file = os.path.splitext(fname)[0]+'.txt'
    json_file = os.path.splitext(fname)[0]+'.json'

    rt = int(os.path.basename(os.path.dirname(fname)))

    _h, _J = load_coef_file(coef_file)
    qbits, energies, spins, occ = load_json_file(json_file)

    N = len(qbits)
    h = [_h[k] for k in qbits]

    qbit_map = {qb:i for i,qb in enumerate(qbits)}
    J = np.zeros([N,N], dtype=float)

    for i, x in _J.items():
        i = qbit_map[i]
        for j, v in x.items():
            j = qbit_map[j]
            J[i,j] = J[j,i] = v

    eps = np.linspace(0,1,nsteps)
    gammas = np.linspace(1,0,nsteps)

    spec = Spectrum()
    spec.solve(h, J, eps, gammas, show=False)
    spec.build_causal(show=False)
    params, pfact = spec.pull_causal(occ, energies)

    append_data(params, rt, pfact)

def run_all(direc, nsteps=100):
    '''Run all problems within the given directory'''

    fnames = get_files(direc)
    np.random.shuffle(fnames)
    for fn in fnames:
        try:
            main(fn, nsteps=nsteps)
        except KeyboardInterrupt:
            continue



if __name__ == '__main__':

    if False:
        try:
            fname = sys.argv[1]
        except KeyError:
            print('Need both sample filename to process...')
            sys.exit()

        main(fname)
    else:
        try:
            dir_name = sys.argv[1]
        except KeyError:
            print('Root directory not specified...')
            sys.exit()

        run_all(dir_name)
