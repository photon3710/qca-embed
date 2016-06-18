import json
from pprint import pprint

class Solution:
    ''' '''

    def __init__(self, fname):
        '''Create a Solution object from a given file'''

        # initialise parameters
        self.qbits = None
        self.fname = fname

        self.load_json(fname)


    def load_json(self, fname):
        '''Load the Solution parameters from a JSON file'''

        try:
            fp = open(fname, 'r')
        except:
            print('Failed to load file: {0}'.format(fname))
            return

        data = json.load(fp)

        # parameters
        self.qbits = data['usedqubits']
        self.gt = data['gt']
        self.hwtime = data['hwtime']
        self.params = data['params']

        # results
        self.spins = data['all_spin']
        self.energies = data['all_energies']
        self.occ = data['all_hwocc']
        self.cell_occ = data['cell_occ']
    
    # accessors
    
    def get_states(self):
        ''' '''
        pass
    
    def get_substate(self, ind):
        ''' '''
        pass