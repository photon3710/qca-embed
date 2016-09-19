#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import sys, re, os

LW = 2
FS = 14
GS = 18

img_dir = '../new_src/img/'
SAVE = True

QCA_OFF = -0.3
QCA_MAX = 2.3

def qca_schedule(X):
    '''Generate a typical QCA latching schedule'''

    N = len(X)
    XX = np.linspace(np.pi/4, 3*np.pi/4, N)

    Delta = QCA_MAX*(1+np.cos(XX))/(1+np.cos(np.pi/4)) + QCA_OFF
    Eps = np.ones([N,], dtype=float)

    return Delta, Eps


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

def main(fname):
    ''' '''

    S, Delta, Eps = load_schedule(fname)

    plt.figure('D-Wave schedule')
    plt.plot(S, Delta, 'g', linewidth=LW)
    plt.plot(S, Eps, 'b', linewidth=LW)
    plt.legend(['$\Delta$(s)', '$\epsilon$(s)'], fontsize=GS)
    plt.xlabel('Annealing parameters (s)', fontsize=FS)
    plt.ylabel('Energy (GHz)', fontsize=FS)

    if SAVE:
        fn = os.path.join(img_dir, 'dwave-schedule.eps')
        plt.savefig(fn, bbox_inches='tight')
    plt.show(block=False)

    # QCA schedule

    Delta, Eps = qca_schedule(S)

    plt.figure('QCA Schedule')
    plt.plot(S, Delta, 'g', linewidth=LW)
    plt.plot(S, Eps, 'b', linewidth=LW)
    plt.legend(['$\Delta$(s)', '$\epsilon$(s)'], fontsize=GS)
    plt.xlabel('Annealing parameters (s)', fontsize=FS)
    plt.ylabel('Energy (dimensionless)', fontsize=FS)

    if SAVE:
        fn = os.path.join(img_dir, 'qca-schedule.eps')
        plt.savefig(fn, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    try:
        fname = sys.argv[1]
    except:
        print('No schedule given...')
        sys.exit()

    main(fname)
