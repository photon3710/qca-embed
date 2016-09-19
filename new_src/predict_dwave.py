#!/usr/bin/env python

import numpy as np

from solvers.spectrum import Spectrum

import sys, os, json, re

from collections import defaultdict
from pprint import pprint

sched_file = '../dat/schedules/Sys4.txt'


def load_schedule(fname):
    ''' '''

    try:
        fp = open(fname, 'r')
    except IOError:
        print('Failed to open schedule file...')
        return

    # re for matching numbers in scientific notation
    regex = re.compile('[+\-]?\d*\.?\d+[eE][+\-]\d+')

    S, Delta, Eps = [], [], []

    for line in fp:
        a, b, c = [float(x) for x in regex.findall(line)]
        S.append(a)
        Delta.append(b/2)
        Eps.append(c)

    fp.close()

    return S, Delta, Eps

def query_schedule(fname, s):
    '''Get the values of the schedule at the parameters s'''

    S, Delta, Eps = load_schedule(fname)

    assert np.min(s) >= 0 and np.max(s)<=1, 'Invalid range for s...'

    delta = np.interp(s, S, Delta)
    eps = np.interp(s, S, Eps)

    return delta, eps

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


def main(fname, nsteps=100):
    ''' '''

    base = os.path.splitext(fname)[0]
    coef_file = base+'.txt'
    json_file = base+'.json'

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

    s = np.linspace(0,1,nsteps)
    gammas, eps = query_schedule(sched_file, s)

    spec = Spectrum()
    spec.solve(h, J, eps, gammas, show=True, exact=True)

    spec.build_causal(show=False)
    spec.compute_eqs()
    pr = spec.predict_probs(rt)


    Es = [eps[-1]*e for e in energies]
    e_probs = spec.format_probs(occ, Es)

    for e in pr:
        print('{0:.4f} :: {1:.4f}:{2:.4f}'.format(e, e_probs[e], pr[e]))



if __name__ == '__main__':

    try:
        fname = sys.argv[1]
    except:
        print('No file base given...')
        sys.exit()

    main(fname)
