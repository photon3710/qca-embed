import json
import numpy as np
#from pprint import pprint
from collections import defaultdict

class Solution:
    ''' '''

    def __init__(self, fname=None):
        '''Create a Solution object from a given file'''

        # initialise parameters
        self.qbits = None
        self.fname = fname

        if fname:
            self.from_json(fname)


    def from_json(self, fname):
        '''Load the Solution parameters from a JSON file'''

        try:
            fp = open(fname, 'r')
        except IOError:
            print('Failed to load file: {0}'.format(fname))
            raise IOError

        data = json.load(fp)
        fp.close()

        # parameters
        self.qbits = [qb[0] for qb in data['usedqubits']]
        self.gt = np.array([zip(*x)[0] for x in data['gt']])
        self.hwtime = data['hwtime']
        self.params = data['params']

        # results
        self.spins = np.array(data['all_spin'], dtype=int).transpose()
        
        self.energies = data['all_energies']
        self.occ = data['all_hwocc']
        self.cell_occ = data['cell_occ']['_ArrayData_']

    def build(self, **kwargs):
        '''Set Solution parameters directly'''
        
        try:
            self.fname = kwargs['fname']
            self.qbits = kwargs['qbits']
            self.gt = kwargs['gt']
            self.hwtime = kwargs['hwtime']
            self.params = kwargs['params']
            self.spins = kwargs['spins']
            self.energies = kwargs['energies']
            self.occ = kwargs['occ']
            self.cell_occ = kwargs['cell_occ']
        except KeyError:
            print('Missing input field')
        
    # accessors
    
    def get_reduced_solution(self, inds, efunc):
        '''Get the states and number of instances for the given indices. Traces
        over other indices'''

        # confirm inds is a subset of qbits
        if not all([ind in self.qbits for ind in inds]):
            raise KeyError('Requested indices invalid...')

        # subset of spins
        qb_map = {self.qbits[i]: i for i in range(len(self.qbits))}
        qb_inds = [qb_map[ind] for ind in inds]
        
        reduced_spins = self.spins[:, qb_inds].tolist()
        
        # construct gt mapping
        gt_map = {}
        gt = {}
        for i in range(self.gt.shape[0]):
            x = tuple(self.gt[i,qb_inds])
            if x not in gt:
                gt[x] = len(gt)+1
            gt_map[i+1] = gt[x]

        gt = np.array(zip(*sorted(gt.items(), key=lambda x:x[1]))[0])
        
        # construct spins mapping
        spin_map = {}
        spins = {}
        for i in range(len(reduced_spins)):
            x = tuple(reduced_spins[i])
            if x not in spins:
                spins[x] = len(spins)+1
            spin_map[i+1] = spins[x]
        
        spins = np.array(zip(*sorted(spins.items(), key=lambda x:x[1]))[0])
        
        # count observed spins for each gauge transformation
        cell_occ = {g: defaultdict(int) for g in range(1, 1+gt.shape[0])}
        occ = defaultdict(int)
        for s, g, o in self.cell_occ:
            cell_occ[gt_map[g]][spin_map[s]] += o
            occ[spin_map[s]] += o

        energies = []
        for spin in spins:
            energies.append(efunc({inds[i]: spin[i] for i in range(len(inds))}))
    
        # sort by energy
        ninds, energies = zip(*sorted(enumerate(energies), key=lambda x: x[1]))
        ind_map = {ninds[k]+1: k+1 for k in range(len(ninds))}
        
        spins = spins[ninds, :]
        occ = [occ[k+1] for k in ninds]
        for g in cell_occ:
            cell_occ[g] = {ind_map[k]: cell_occ[g][k] for k in cell_occ[g]}
        
        new_sol = Solution()
        kwargs = {'fname': self.fname,
                  'qbits': inds,
                  'gt': gt,
                  'hwtime': self.hwtime,
                  'params': self.params,
                  'spins': spins,
                  'energies': energies,
                  'occ': occ,
                  'cell_occ': cell_occ}
        new_sol.build(**kwargs)

        return new_sol