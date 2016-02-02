#!/usr/bin/env python

#---------------------------------------------------------
# Name: v_cell.py
# Purpose: Simulation of v-cell type QCA
# Author: Jacob Retallick
# Created: 2015.10.28
# Last Modified: 2015.10.28
#-----------------------------------------------

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import sys


# constants
FS = 16
SWEEP_E = False

def factor(eps, g1, g2):
    '''Compute the polarization factor'''

    g2 = 0
    return abs(eps)/np.sqrt(g1*g1+eps*eps + 2*g2*g2)    

def ground_state(eps, g1, g2, E0):
    '''construct a single cell hamiltonian with net interaction term E and
    pol-pol and pol-null tunnelling parameters of g1 and g2'''

    H = np.matrix([[eps, -g1, -g2],
                   [-g1, -eps, -g2],
                   [-g2, -g2, -E0]])
    vals, vecs = np.linalg.eigh(H)
  

    return vals[0], vecs[:, 0]

def get_lam0(eps, g1, gam0):
    ''' '''
    
    g2, lam0, grnd = ground_state(eps, g1, gam0, 0)
    
    print('{0:.4f}: {1:.4f}'.format(lam0, -np.sqrt(eps*eps+g1*g1+2*gam0*gam0)))

    return g2, lam0
    
P = np.asmatrix(np.diag([-1, 1, 0]))


def state_pol(state):
    '''calculate the cell polarization of a state'''

    st = np.asmatrix(state)
    return float(st.getH()*P*st)


N = np.asmatrix(np.diag([1, 1, 0]))


def state_pop(state):
    '''calculate the cell population of the state'''

    st = np.asmatrix(state)
    return float(st.getH()*N*st)


def eps_sweep(E, eps_max, g1, g2, res=20):
    '''Sweep pol energy for a given set of gamma values'''
    
    Es = np.linspace(-eps_max, eps_max, res)
    
    lams = []
    pols = []
    pops = []
    
    for eps in Es:
        lam, grnd = ground_state(eps, g1, g2, E)
        lams.append(lam)
        pols.append(state_pol(grnd))
        pops.append(state_pop(grnd))
    
    fact = factor(eps, g1, g2)
    plt.figure('Pol and Pop')
    plt.plot(Es, np.array(pols)/fact, 'r', linewidth=2)
    plt.plot(Es, pops, 'b--', linewidth=2)
    plt.xlabel('$\epsilon$', fontsize=FS)
    plt.title('Cell Polarization and Population', fontsize=FS)
    plt.legend(['Polarizations', 'Population'])
    plt.show(block=False)
    
#    plt.figure('Ground state energy')    
#    plt.plot(E0s, np.array(lams), 'r')
#    plt.xlabel('$E_0$', fontsize=FS)
#    plt.ylabel('$E_{grnd}$', fontsize=FS)
#    plt.title('Ground state energy', fontsize=FS)
#    plt.show(block=False)
#    
#    plt.figure('Polarization factor')
#    plt.plot(E0s, np.array(pols)/np.array(pops), 'g', linewidth=2)
#    plt.axhline(factor(eps, g1, g2))
#    plt.xlabel('$E_0$', fontsize=FS)
#    plt.ylabel('P/N', fontsize=FS)
#    plt.title('Polarization factor', fontsize=FS)
#    plt.show(block=False)
    
    plt.show(block=True)
    

def E_sweep(eps, E_max, g1, g2, res=20):
    '''Sweep null-site energy for a given set of gamma values'''

    E0s = np.linspace(-E_max, E_max, res)
    
    lams = []
    pols = []
    pops = []
    
    for E0 in E0s:
        lam, grnd = ground_state(eps, g1, g2, E0)
        lams.append(lam)
        pols.append(state_pol(grnd))
        pops.append(state_pop(grnd))
    
    fact = factor(eps, g1, g2)
    plt.figure('Pol and Pop')
    plt.plot(E0s, np.array(pols)/fact, 'r', linewidth=2)
    plt.plot(E0s, pops, 'b--', linewidth=2)
    plt.xlabel('$E_0$', fontsize=FS)
    plt.title('Cell Polarization and Population', fontsize=FS)
    plt.legend(['Polarizations', 'Population'])
    plt.show(block=False)
    
#    plt.figure('Ground state energy')    
#    plt.plot(E0s, np.array(lams), 'r')
#    plt.xlabel('$E_0$', fontsize=FS)
#    plt.ylabel('$E_{grnd}$', fontsize=FS)
#    plt.title('Ground state energy', fontsize=FS)
#    plt.show(block=False)
#    
#    plt.figure('Polarization factor')
#    plt.plot(E0s, np.array(pols)/np.array(pops), 'g', linewidth=2)
#    plt.axhline(factor(eps, g1, g2))
#    plt.xlabel('$E_0$', fontsize=FS)
#    plt.ylabel('P/N', fontsize=FS)
#    plt.title('Polarization factor', fontsize=FS)
#    plt.show(block=False)
    
    plt.show(block=True)


if __name__ == '__main__':

    try:
        eps, E_max, g1, g2 = [float(x) for x in sys.argv[1:5]]
    except:
        print('Insufficient arguments given...')
        sys.exit()
        
    try:
        res = int(sys.argv[5])
    except:
        res = 20
    
    eps_sweep(E_max, eps, g1, g2, res=res)
    E_sweep(eps, E_max, g1, g2, res=res)
    
    

