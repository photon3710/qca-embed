from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from pprint import pprint

from parse_qca import parse_qca_file

import sys

GMIN = 0.1      # weak tunneling limit
GMAX = 10       # strong tunneling limit
NSTEPS = 400     # nmber of steps between clocks

T = 273               # characteristic temperature
kT = 8.617e-5*T     # characteristic energy

# formatting
FS = 14
    
def gamma_schedule(gmin, gmax, N, nsteps, offset=0.):
    '''Clocking schedule for N zones with nsteps between clocks'''
    
    for step in xrange(nsteps):
        yield [gmin+.5*(gmax-gmin)*(1-np.cos(2*np.pi*(step/nsteps - \
            (n%4)/4+offset))) for n in xrange(N)]

def sorted_merge(lists):
    return sorted(reduce(lambda x, y: x+y, lists))
    
def construct_coefs(cells, zones, J):
    ''' '''
    
    # collapse each zone to a single set of cells
    zones = [sorted_merge(zone) for zone in zones]
    
    # compute h parameters
    h = np.array([cell['pol'] if 'pol' in cell else 0 for cell in cells])
    h = np.dot(J, h)
    
    # identify driver/fixed cells
    drivers = [i for i in xrange(len(cells)) if 'pol' in cells[i]]
    
    # remove driver cell indices from zones
    for d in drivers:
        for z in zones:
            if d in z:
                z.remove(d)
                break
    
    # isolate zone h arrays
    hs = [h[z] for z in zones]
    
    # isolate zone J arrays
    Js = [J[z, :][:, z] for z in zones]
    
    # isolate inter zone coupling arrays
    Jxs = {(i, i+1): J[zones[i], :][:, zones[i+1]] for i in xrange(len(zones)-1)}
    
    return hs, Js, Jxs
    
# pauli matrix generators
    
sz = np.array([-1, 1])
sx = sp.dia_matrix([[0, 1], [1, 0]], dtype=int)
eye = sp.eye(2, dtype=int)

def pauli_z(n, N):
    '''compute diag(pauli_z) array for n in N'''
    
    return np.tile(np.repeat(sz, 2**(N-n)), [1, 2**(n-1)])

def to_state(vecs):
    '''map eig vecs to state polarizations'''
    N = int(np.log2(vecs.shape[0]))
    probs = np.power(np.abs(vecs), 2)
    
    D = [np.array(np.dot(pauli_z(i, N), probs)) for i in range(1, N+1)]
    D = np.array(D)
    
    print(pprint(np.round(D, 2)))
    
def gen_gamma(N):
    '''compute sum of tunneling term for a single zone'''
    
    if N==1:
        return sx
    
    return sp.kron(gen_gamma(N-1), eye) + sp.kron(sp.eye(2**(N-1)), sx)

# Hamiltonian generation

def construct_H0(h, J):
    '''Construct on-site classical Hamiltonian of a single zone'''

    N = len(h)
    
    # precompute pauli z
    PZ = [pauli_z(n, N) for n in xrange(1, N+1)]

    H = np.zeros([1, 2**(N)], dtype=float)
    
    for i in xrange(N):
        if h[i] != 0:
            H += h[i]*PZ[i]
        for j in xrange(i+1, N):
            if J[i, j] != 0:
                H += J[i, j]*PZ[i]*PZ[j]
    
    return H


def construct_hams(hs, Js, Jxs):
    '''Construct static Hamiltonians'''
    
    Nz = len(hs)     # number of zones
    
    H0s = [construct_H0(hs[i], Js[i]) for i in xrange(Nz)]
    Gammas = [gen_gamma(len(hs[i])) for i in xrange(Nz)]

    M = [H.size for H in H0s]   # number of states per zone
    C = [1]+list(np.cumprod(M))     # cumprod of H0 sizes

    # pad single zone Hamiltonians
    for i in xrange(Nz):
        H0s[i] = np.tile(np.repeat(H0s[i], C[-1]/C[i+1]), [1, C[i]])
        Gammas[i] = sp.kron(sp.eye(C[i]), sp.kron(Gammas[i], sp.eye(C[-1]/C[i+1])))
    
    # only need sum of H0s
    H0 = np.sum(np.array(H0s), axis=0)
    
    # add zone-zone interactions to H0
    for i,j in Jxs:
        Jx = Jxs[(i,j)]
        Hx = np.zeros([1, M[i]*M[j]], dtype=float)
        for n,m in zip(*Jx.nonzero()):
            Hx += Jx[n, m]*np.kron(pauli_z(n+1, Jx.shape[0]),
                                   pauli_z(m+1, Jx.shape[1]))
        if np.any(Hx):
            H0 += np.tile(np.repeat(Hx, C[i]), [1, C[-1]/C[j+1]])
    
    H0 = sp.diags(H0, [0])
    return H0, Gammas
    
# spectrum decomposition

def factor(E):
    return np.exp(-E/kT)

def state_substate(ind, Mz):
    '''Decompose a composite state index into sub-space indices'''
    
    # get binary representation of index
    b = format(ind, '#0{0}b'.format(sum(Mz)+2))[2::]
    
    # split binary into sub-indices
    bs = []
    for M in Mz:
        temp, b = b[0:M], b[M::]
        bs.append(temp)

    # convert sub-indices to decimal
    inds = [int(_, 2) for _ in bs]
    
    return inds
    
def decompose(vecs, vals, Mz):
    '''Collapse state space to mode decomp'''
    
    # comp: state amplitudes scaled by Boltzmann factor
    # scomp: comp for each zone

    Es = np.array(vals)-np.min(vals)
    
    facts = factor(Es)
    amps = np.power(np.abs(vecs), 2)
    comp = np.asarray(np.dot(amps, facts)).reshape([-1,])
    
    # break into sub-space modes
    scomps = [[0]*2**M for M in Mz]
    for ind in range(vecs.shape[0]):
        inds = state_substate(ind, Mz)
        for i in xrange(len(Mz)):
            scomps[i][inds[i]] += comp[ind]

    return comp, scomps
    
def main(fname):
    ''' '''
    
    plt.close('all')

    # parse QCA file
    cells, spacing, zones, J, feedback = parse_qca_file(fname, one_zone=False)
    J /= -np.max(np.abs(J))

    Nz = len(zones)     # number of zones

    # get isolated coefficients
    hs, Js, Jxs = construct_coefs(cells, zones, J)
    Mz = [len(h) for h in hs]    

    # construct static Hamiltonians
    H0, Gammas = construct_hams(hs, Js, Jxs)
    
    # loop through gamma configurations
    schedule = gamma_schedule(GMIN, GMAX, Nz, NSTEPS)
    G = []
    C = []
    SC = [[] for _ in xrange(Nz)]
    E = []
    while True:
        try:
            gammas = next(schedule)
        except:
            break
        sys.stdout.write('\r{0:.1f}%'.format((len(G)+1)*100./NSTEPS))
        sys.stdout.flush()
        G.append(gammas)

        # construct new Hamiltonian
        H = H0.copy()
        for n in xrange(Nz):
            H = H + gammas[n]*Gammas[n]
        
        # analyse spectrum
        if H.shape[0]>5:
            evals, evecs = eigsh(H, which='SA', k=2*int(np.log2(H.shape[0])))
        else:
            evals, evecs = eigh(H.todense())
        
#        to_state(evecs)
        
        # decompose and store
        comp, scomps = decompose(evecs, evals, Mz)
        
        C.append(comp)
        for i in xrange(Nz):
            SC[i].append(scomps[i])
        E.append(evals)
    
    # standardize formatting
    G = np.array(G)
    E = np.array(E)
    C = np.array(C).transpose()
    SC = [np.array(SC[i]).transpose() for i in xrange(Nz)]

    # find state reordering, sorted by max contribution summed over full clock
    C = C[np.argsort(-np.sum(C, axis=1)),:]
    SC = [sc[np.argsort(-np.sum(sc, axis=1)), :] for sc in SC]    
    
    C = C[0:int(np.sqrt(C.shape[0])), :]
#    SC = [sc[0:int(np.sqrt(sc.shape[0])), :] for sc in SC]

    X = np.linspace(0, 1, NSTEPS)
    
    plt.figure('Spectrum')
    plt.plot(X, E)
    plt.xlabel('Time', fontsize=FS)
    plt.ylabel('Energy (eV)', fontsize=FS)
    plt.show(block=False)
    
    plt.figure('Gamma Schedule')
    plt.plot(X, G)
    plt.xlabel('Time', fontsize=FS)
    plt.ylabel('Gamma', fontsize=FS)
    plt.show(block=False)
    
    plt.figure('Decomposition')
    plt.imshow(C, interpolation='none', aspect='auto', origin='lower')
    plt.xlabel('Time', fontsize=FS)
    plt.ylabel('State index', fontsize=FS)
    plt.colorbar()
    plt.show(block=False)
    
    for i in xrange(Nz):
        plt.figure('Zone: {0}'.format(i))
        plt.imshow(SC[i], interpolation='none', aspect='auto', origin='lower')
        plt.xlabel('Time', fontsize=FS)
        plt.ylabel('State index', fontsize=FS)
        plt.colorbar()
        plt.show(block=False)
    
    plt.show(block=True)
    print('\n')


if __name__ == '__main__':
    
    try:
        fname = sys.argv[1]
    except:
        print('No QCA file given...')
        sys.exit()
    
    main(fname)
