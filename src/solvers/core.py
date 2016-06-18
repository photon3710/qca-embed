#!/usr/bin/env python
import numpy as np
import scipy.sparse as sp


# 
def pauli_z(n, N, pz=[1,-1]):
    '''compute diag(pauli_z) for the n^th of N spins (n in [1...N]). Can
    override '''
    if n < 1 or n > N:
        print('Invalid spin index')
        return None
    return np.tile(np.repeat(pz, 2**(N-n)), [1, 2**(n-1)])
