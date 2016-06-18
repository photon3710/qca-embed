from PyQt4 import QtGui, QtCore
from core.dwave_sol import DWAVE_Sol

import sys, os, re
import numpy as np

import matplotlib.pyplot as plt


RES_DIR = os.path.join('solutions','results', 'cross-talk')
RES_DIR = os.path.join(os.getcwd(), os.pardir, RES_DIR) 

COEF_DIR = os.path.join('experiments', 'cross-talk')
COEF_DIR = os.path.join(os.getcwd(), os.pardir, COEF_DIR) 

E_TOL = 1e-6    # allowed difference between given and computed energies
N_THRESH = 10
        
FS = 14

def modified_kldiv(P, Q):
    '''Compute the kl divergence between a "true" distribution P and some other
    distribution Q with Q=0 -> no contribution'''
    
    assert len(P) == len(Q), 'Distribution of different lengths'

    P = np.array(P).reshape([-1,])
    Q = np.array(Q).reshape([-1,])
    
    pind = np.nonzero(P)[0]
    qind = np.nonzero(Q)[0]
    inds = np.intersect1d(pind, qind)
    
    print(np.sum(P[inds]), np.sum(Q[inds]))
    return np.sum(P[inds]*np.log(P[inds])) - np.sum(P[inds]*np.log(Q[inds]))

def total_var_dist(P, Q):
    P = np.array(P).reshape([-1,])
    Q = np.array(Q).reshape([-1,])
    return .5*np.sum(np.abs(P-Q))
    
def hellinger_dist(P, Q):
    P = np.array(P).reshape([-1,])
    Q = np.array(Q).reshape([-1,])
    return np.sum((np.sqrt(P)-np.sqrt(Q))**2)/np.sqrt(2)
    
stat_dist = total_var_dist

def compute_E(h, J, spins, r=4):
    '''compute the energy of a given state, all parameters are dicts'''
    
    E = 0.
    
    try:
        for i in spins:
            if i in h:
                E += h[i]*spins[i]
            if i in J:
                for j in J[i]:
                    if j in spins:
                        E += J[i][j]*spins[i]*spins[j]
    except KeyError:
        print('Mismatch in spin indices...')
        return None

    return np.round(E, r)

class HLine(QtGui.QFrame):
    ''' '''
    
    def __init__(self, parent=None):
        super(HLine, self).__init__(parent)
        self.setFrameStyle(QtGui.QFrame.HLine)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)

class MainWindow(QtGui.QMainWindow):
    ''' '''
    
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()
        self.initUI()
        
    def initUI(self):
        ''' '''
        
        # initial settings
        self.parent_dir_base = os.getcwd()
        self.result_dir_base = RES_DIR
        self.coef_dir_base = COEF_DIR
        
        self.n_thresh = N_THRESH
        
        # main window parameters
        geo = [100, 100, 400, 100]
        self.setGeometry(*geo)
        self.setWindowTitle('Cross Talk Analysis')
        
        vbox = QtGui.QVBoxLayout()
        
        # AB File line
        
        hb1 = QtGui.QHBoxLayout()
        
        l1 = QtGui.QLabel()
        l1.setText('AB File:')
        
        self.i1 = QtGui.QLineEdit(self)
        b1 = QtGui.QPushButton(self, text='...')
        b1.clicked.connect(self.browse_AB_file)
        
        hb1.addWidget(l1, stretch=0)
        hb1.addWidget(self.i1, stretch=5)
        hb1.addWidget(b1, stretch=1)
        
        # coef file line
        
        hb2 = QtGui.QHBoxLayout()
        
        l2 = QtGui.QLabel()
        l2.setText('Coef file:')
        
        self.i2 = QtGui.QLineEdit(self)
        b2 = QtGui.QPushButton(self, text='...')
        b2.clicked.connect(self.browse_coef_file)
        
        hb2.addWidget(l2, stretch=0)
        hb2.addWidget(self.i2, stretch=5)
        hb2.addWidget(b2, stretch=1)
        
        # Command buttons
        
        hb3 = QtGui.QHBoxLayout()
        
        b3 = QtGui.QPushButton(self, text='Process')
        b3.clicked.connect(self.run_processing)
        
        b4 = QtGui.QPushButton(self, text='Exit')
        b4.clicked.connect(QtCore.QCoreApplication.instance().quit)
        
        self.i3 = QtGui.QLineEdit(self)
        self.i3.setText(str(N_THRESH))
    
        hb3.addWidget(self.i3, stretch=3)
        hb3.addWidget(b3, stretch=1)
        hb3.addWidget(b4, stretch=1)
        
        vbox.addLayout(hb1)
        vbox.addLayout(hb2)        
        vbox.addWidget(HLine())
        vbox.addLayout(hb3)
        
        # central widget
        cw = QtGui.QWidget(self)
        cw.setLayout(vbox)
        self.setCentralWidget(cw)
        
    def browse_AB_file(self):
        ''' '''
        
        ab_fname = str(QtGui.QFileDialog.getOpenFileName(
            self, 'Select AB File...', self.result_dir_base))
        
        if ab_fname:
            self.result_dir_base = os.path.dirname(ab_fname)
            self.i1.setText(ab_fname)
    
    def browse_coef_file(self):
        ''' '''
        coef_fname = str(QtGui.QFileDialog.getOpenFileName(
            self, 'Select Coef File...', self.coef_dir_base))
            
        if coef_fname:
            self.coef_dir_base = os.path.dirname(coef_fname)
            self.i2.setText(coef_fname)

    def load_coef_file(self, fname):
        ''' '''
        
        try:
            fp = open(fname, 'r')
        except IOError:
            print('Failed to load file: {0}'.format(fname))
            return None
        
        nqbits = int(fp.readline())
        print('Loading coef file with {0} qbits'.format(nqbits))
        
        h = {}
        J = {}
        for line in fp:
            a, b, v = line.split()
            a, b, v = int(a), int(b), float(v)
            if a==b:
                h[a] = v
            else:
                if a in J:
                    J[a][b] = v
                else:
                    J[a] = {b: v}
        fp.close()

        return h, J

    def dist_comp(self, P, Q, label='', block=True, plot=False):
        '''Direct comparison of two Solution distribution'''
        
        hash_ = lambda s: hash(tuple(s.tolist()))
        
        print('Distribution comparison:')
        print('\tP spins: {0}'.format(P.spins.shape))
        print('\tQ spins: {0}'.format(Q.spins.shape))
        
        assert P.spins.shape[1] == Q.spins.shape[1], \
            'Solutions have different numbers of qubits'
        
        # common dict of state hashes and energies
        energies = {}
        for i in range(len(P.energies)):
            key, E = hash_(P.spins[i, :]), P.energies[i]
            energies[key] = E
        for i in range(len(Q.energies)):
            key, E = hash_(Q.spins[i, :]), Q.energies[i]
            if key in energies and abs(energies[key]-E) > E_TOL:
                print('')
            energies[key] = E

        key_inds = {k: i for i, k in \
            enumerate(sorted(energies, key=lambda x: energies[x]))}
        occs = np.zeros([2, len(energies)], dtype=int)
        
        for i in range(P.spins.shape[0]):
            occs[0, key_inds[hash_(P.spins[i, :])]] = P.occ[i]
        for i in range(Q.spins.shape[0]):
            occs[1, key_inds[hash_(Q.spins[i, :])]] = Q.occ[i]

        # remove rare outputs
        if True:
            inds = np.nonzero(np.sum(occs, axis=0) > self.n_thresh)[0]
            occs = occs[:, inds]
            
        # compute statistical distance
        pdfs = [occs[i,:]*1./np.sum(occs[i,:]) for i in [0,1]]
        sd = stat_dist(*pdfs)
        
        print('Statistical Distance: {0:.3f}'.format(sd))
        
        if plot:
            # plot both distributions
            X = np.arange(occs.shape[1])
            w = .3
            plt.figure(label)
            plt.bar(X-.75*w, occs[0, :], width=.5*w, color=(1, 0, 0))
            plt.bar(X+.25*w, occs[1, :], width=.5*w, color=(0, 0, 1))
            plt.title('Number of output occurances', fontsize=FS)
            plt.legend([label, 'AB'], fontsize=FS)
            plt.xlabel('State index', fontsize=FS)
            plt.ylabel('Number of occurances', fontsize=FS)
            plt.show(block=block)
        
        
    def run_processing(self):
        ''' '''
        
        ab_fname = os.path.normpath(str(self.i1.text()))
        coef_fname = os.path.normpath(str(self.i2.text()))
        self.n_thresh = int(self.i3.text())
        
        # confirm fname  format
        if not re.match('.*_AB_[0-9]+us.json', ab_fname):
            print('Invalid filename format...')
            return
            
        if not coef_fname:
            print('Missing coef file')
            return
    
        root, _, ext = ab_fname.rpartition('AB')
        
#        rt = re.search('[0-9]+', ext).group(0)
        
        # not very robust but good enough or now
        a_fname = root+'A'+ext
        b_fname = root+'B'+ext
        
        # check that all solution files exist
        if not all([os.path.exists(fn) for fn in [a_fname, b_fname, ab_fname]]):
            print('Missing filenames...')
            return
        
        # load coef file
        h, J = self.load_coef_file(coef_fname)
        e_func = lambda qb_spins: compute_E(h, J, qb_spins)
        # preamble done, now process
        print('loading solution files...')
        try:
            a_sol = DWAVE_Sol(a_fname)
            b_sol = DWAVE_Sol(b_fname)
            ab_sol = DWAVE_Sol(ab_fname)
        except IOError:
            print('Failed to read at least one of the solution files')
            return

        # get marginal distribution of ab_so
        ab_marg = {}
        
        print('getting marginalised solution for problem A...')
        ab_marg['A'] = ab_sol.get_reduced_solution(a_sol.qbits, e_func)
        
        print('getting marginalised solution for problem B...')
        ab_marg['B'] = ab_sol.get_reduced_solution(b_sol.qbits, e_func)
        
        # check independence of A and B
        print('checking independence of A and B...')
        hash_ = lambda s: hash(tuple(s.tolist()))
        qb_map = {k: i for i, k in \
            enumerate(sorted(ab_marg['A'].qbits + ab_marg['B'].qbits))}
        a_inds = [qb_map[qb] for qb in ab_marg['A'].qbits]
        b_inds = [qb_map[qb] for qb in ab_marg['B'].qbits]
        
        a_keys, a_key_inds, i = {}, [], 0
        for j in range(len(ab_marg['A'].energies)):
            key, occ = hash_(ab_marg['A'].spins[j, :]), ab_marg['A'].occ[j]
            if occ > self.n_thresh:
                a_keys[key], i = i, i+1
                a_key_inds.append(j)

        b_keys, b_key_inds, i = {}, [], 0
        for j in range(len(ab_marg['B'].energies)):
            key, occ = hash_(ab_marg['B'].spins[j, :]), ab_marg['B'].occ[j]
            if occ > self.n_thresh:
                b_keys[key], i = i, i+1
                b_key_inds.append(j)
        
        if len(a_keys)*len(b_keys) < 1e5:
            ab_occ = np.zeros([len(a_keys), len(b_keys)], dtype=int)
            for i in range(ab_sol.spins.shape[0]):
                spin = ab_sol.spins[i, :]
                ka, kb = hash_(spin[a_inds]), hash_(spin[b_inds])
                if ka in a_keys and kb in b_keys:
                    ab_occ[a_keys[ka], b_keys[kb]] = ab_sol.occ[i]
            ab_marg_occ = np.outer(np.array(ab_marg['A'].occ)[a_key_inds], 
                                   np.array(ab_marg['B'].occ)[b_key_inds])
            
            print(ab_occ.shape)
            print(ab_marg_occ.shape)
            
            ab_occ = ab_occ*1./np.sum(ab_sol.occ)
            ab_marg_occ = ab_marg_occ*1./(np.sum(a_sol.occ)*np.sum(b_sol.occ))
            
            # statistical distance
            ab_sd = stat_dist(ab_occ, ab_marg_occ)
            print('Statistical Distance: {0:.4f}'.format(ab_sd))

            if True:
                
                vmax = max(np.max(ab_occ), np.max(ab_marg_occ))
                plt.figure('JD')
                plt.imshow(ab_occ, interpolation='none', aspect='auto', vmin=0, vmax=vmax)
                plt.colorbar()
                plt.title('Joint distribution', fontsize=FS)
                plt.xlabel('A state index', fontsize=FS)
                plt.ylabel('B state index', fontsize=FS)
                plt.show(block=False)
                
                plt.figure('JMD')
                plt.imshow(ab_marg_occ, interpolation='none', aspect='auto', vmin=0, vmax=vmax)
                plt.colorbar()
                plt.title('Joint marginal distribution', fontsize=FS)
                plt.xlabel('A state index', fontsize=FS)
                plt.ylabel('B state index', fontsize=FS)
                plt.show(block=True)
                
#                plt.figure('JD-Diff')
#                plt.imshow(np.abs(ab_occ-ab_marg_occ), interpolation='none', aspect='auto')
#                plt.colorbar()
#                plt.title('Joint distribution diff', fontsize=FS)
#                plt.xlabel('A state index', fontsize=FS)
#                plt.ylabel('B state index', fontsize=FS)
#                plt.show(block=True)
        else:
            print('Too many data points to safely plot')
        
        # compare marginal distribution to isolated distribution
        self.dist_comp(a_sol, ab_marg['A'], 'A', plot=True)
        self.dist_comp(b_sol, ab_marg['B'], 'B', plot=True)

        
        
def main():
    '''Main loop which initialises application'''
    
    app = QtGui.QApplication(sys.argv)
    
    w = MainWindow()
    w.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()