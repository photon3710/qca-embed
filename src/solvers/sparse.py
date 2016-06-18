import numpy as np
import scipy.sparse as sp
from base_class import Solver

class SparseSolver(Solver):
    '''Eigenvalue solver using sparse matrix formulation'''
    
    def __init__(self):
        ''' '''
        super(self, SparseSolver).__init__()
        
        