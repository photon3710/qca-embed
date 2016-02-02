#!/usr/bin/env python

#---------------------------------------------------------
# Name: qca_anneal.py
# Purpose: Look at state evolution in clocking of QCA circuits
# Author: Jacob Retallick
# Created: 2015.10.28
# Last Modified: 2015.10.28
#-----------------------------------------------


from solve import solve
import numpy as np
from matplotlib import pyplot as plt
from auxil import stateToPol

DUAL = False
FS = 16

SAVE = False
ROOT = 'maj_'


def echo_comp(vec):
    '''echo the primary components of a vector'''
    comp = np.argsort(vec).tolist()[::-1]
#    print stateToPol(vec)
    print '\n'+'\n'.join(bin(x)[2::] for x in comp[:5])


def anneal(h, J, g0, steps=50):
    '''Get ground state configuration of a circuit specified by h and J for
    gammas values from g_max to g_min'''

    time = np.linspace(0, 1, steps)
    spectrum = []
    ground = []
    excited = []
    for i in xrange(0, steps):
        fact = (i+1)*1./(steps+1)
        gamma = g0*(1-fact)
        if DUAL:
            _h, _J = fact*h, fact*J
        else:
            _h, _J = h, J
        gs, es, spec, sols = solve(_h, _J, gamma=gamma, full_output=True,
                                   minimal=False, exact=False)
        spectrum.append(np.array(spec[0::]))
        ground.append(np.abs(np.array(sols['eigsh']['vecs'][:, 0])))
        excited.append(np.abs(np.array(sols['eigsh']['vecs'][:, 1])))

    spectrum = np.array(spectrum)
    ground = np.array(ground)
    excited = np.array(excited)
    
    # identify ground state components
#    echo_comp(ground[int(.8*steps)])
#    echo_comp(excited[int(.8*steps)])
#    echo_comp(excited[int(.9*steps)])

    plt.figure('Spectrum')
    plt.clf()
    plt.plot(time, spectrum, linewidth=2)
    plt.xlabel('Time', fontsize=FS)
    plt.ylabel('Energy', fontsize=FS)
    plt.title('Low Energy Spectrum', fontsize=FS)
    plt.show(block=False)
    if SAVE:
        plt.savefig(ROOT+'spec.eps', bbox_inches='tight')

    plt.figure('Ground State Configuration')
    plt.clf()
    plt.plot(time, np.power(ground, 2), linewidth=2)
    plt.xlabel('Time', fontsize=FS)
    plt.ylabel('Basis state probabilities', fontsize=FS)
    plt.title('Ground state configuration', fontsize=FS)
    plt.show()
    if SAVE:
        plt.savefig(ROOT+'gr.eps', bbox_inches='tight')

    if True:
        plt.figure('Excited State Configuration')
        plt.clf()
        plt.plot(time, np.power(excited, 2), linewidth=2)
        plt.xlabel('Time', fontsize=FS)
        plt.ylabel('Basis state probabilities', fontsize=FS)
        plt.title('Excited State Configuration', fontsize=FS)
        plt.show()
        if SAVE:
            plt.savefig(ROOT+'ex.eps', bbox_inches='tight')


def maj_coef(P):
    h = np.array([P[0], P[1], P[2], 0, 0])
    J = np.array([[0, -.2, 0, 1, -.2],
                  [-.2, 0, -.2, 1, 0],
                  [0, -.2, 0, 1, -.2],
                  [1, 1, 1, 0, 1],
                  [-.2, 0, -.2, 1, 0]])
    return h, J


def inv_coef():
    h = np.array([1, 0, 0, 0, 0, 0, 0])
    J = np.array([[0, 1, -.2, 0, -.2, 0, 0],
                  [1, 0, 1, -.2, 1, -.2, 0],
                  [-.2, 1, 0, 1, 0, 0, 0],
                  [0, -.2, 1, 0, 0, 0, -.2],
                  [-.2, 1, 0, 0, 0, 1, 0],
                  [0, -.2, 0, 0, 1, 0, -.2],
                  [0, 0, 0, -.2, 0, -.2, 0]])
    return h, J


if __name__ == '__main__':

    h = np.array([1, 0, 0])
    C = [-1, 1]
    J = np.array([[0, C[0], 0],
                  [C[0], 0, C[1]],
                  [0, C[1], 0]])

    h, J = maj_coef([1, -1, 1])
#    h, J = inv_coef()
    anneal(h, -J, 2, 200)
