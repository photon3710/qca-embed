#!/usr/bin/end python


import numpy as np
from numpy.linalg import eigh

import matplotlib.pyplot as plt
import solvers.core as core

import sys

pauli_x = np.array([[0,1],[1,0]])
H_density = 0.4

def gen_hamiltonian(N,d=H_density):

    h = 2*np.random.rand(N)-1
    J = np.triu(2*np.random.rand(N,N)-1,1)

    J = J*(np.abs(J)<d)
    J += J.T

    Hx = core.generate_Hx(np.ones([N,]))
    Hz = core.generate_Hz(h, J)

    return Hx, Hz

def compute_utep(H_I, H_P, s, gammas, epsilon):

    G = np.zeros(H_P.shape, dtype=float)

    ds = s[1]-s[0]
    for gam, eps in zip(gammas, epsilon):
        H = -gam+H_I+eps*H_P

def main(nsteps=100, N=2):
    ''' '''

    H_I, H_P = gen_hamiltonian(N)

    print(sorted(np.diag(H_P)))

    s = np.linspace(0,1,nsteps)
    gammas = np.linspace(1,0,nsteps)
    epsilon = np.linspace(0,1,nsteps)

    G = np.zeros([2**N, 2**N], dtype=float) # utep generator

    spectrum = []
    for gam, eps in zip(gammas, epsilon):
        H = -gam*H_I+eps*H_P
        evals, evecs = eigh(H)
        spectrum.append(evals)
        G += H

    plt.plot(s, spectrum)
    plt.show()


if __name__ == '__main__':

    try:
        N = int(sys.argv[1])
    except:
        N = 2

    main(100, N)
