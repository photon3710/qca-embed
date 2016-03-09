from PyQt4 import QtGui, QtCore
from core.solution import Solution
from core.matrix_seriation import seriate

import sys, os, re

from pprint import pprint
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
import numpy as np
from random import sample

FS = 14
RES_DIR = 'solutions'
RES_DIR = os.path.join(os.getcwd(), os.pardir, RES_DIR) 

COEF_DIR = 'experiments'
COEF_DIR = os.path.join(os.getcwd(), os.pardir, COEF_DIR) 


def color_gen(N, s=1., v=1.):
    '''Generate N distinct colors with equal luminance'''
    
    nn = range(N)
    np.random.shuffle(nn)
    for n in nn:
        yield(hsv_to_rgb(n*1./N, s, v))
        
def stat_dist(A, B, typ='TV'):
    '''Compute the statistic distance between the two pdfs A and B'''
    
    # normalise
    A_ = A*1./np.sum(A)
    B_ = B*1./np.sum(B)
    
    # total variation
    if typ=='TV':
        return .5*np.sum(np.abs(A_-B_))
        
def match_dist(D, targ, K):
    '''Estimate avarage statistical distance between target disitrbution and
    averaged choices of K distributions from D '''
    
    sd_av = 0
    delta = 1.
    
    M = D.shape[0]
    SDs = {}
    
    n = 0
    
    while abs(delta)>1e-4:
        # randomly generate pair of indices
        c = 0 # max number of attempts for new indices
        while True and c < 1000:
            inds = sorted(sample(range(M), K))
            if tuple(inds) not in SDs:
                break
            c += 1
        # update sd_av estimate witth generated indices
        d = np.sum(D[inds,:], axis=0)*1./np.sum(D[inds,:])
        sd = stat_dist(d, targ)
        SDs[tuple(inds)] = sd
        delta = (sd-sd_av)*1./(n+1)
        sd_av += delta
        n += 1

    var = sum((sd-sd_av)**2 for sd in SDs.values())*1./(n-1)
    return sd_av, np.sqrt(var)
    
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
    

class stackPlot:
    ''' '''
    
    def __init__(self, label=None):
        
        assert label is not None, 'Must specify figure label'
        
        self.label = label
        self.w = 1.
        
    def set_data(self, data):
        '''Set a 2D set of non-negative data'''
        if np.min(data) < 0:
            print('Must provide non-negative data')
            return
        self.data = data
            
    def set_format(self):
        ''' '''
        pass
            
    def plot(self, block=True):
        ''' '''
        
        self.fig = plt.figure(self.label)
        plt.clf()
        
        M, N = self.data.shape
        
        cg = color_gen(M)

        B = np.zeros(N, dtype=float)
        x = range(N)
        
        print('{0} gauge transformations'.format(M))
        print('{0} observed states'.format(N))

        n0 = 3
        for i in range(n0, n0+2):
            d = self.data[i,:]
            plt.bar(x, d, self.w, color=next(cg), bottom=B)
            B += d
            # progress bar
            sys.stdout.write('\rProgress: {0}%'.format((i+1)*100./M))
            sys.stdout.flush()
        
        plt.show(block=block)
        
    
class HLine(QtGui.QFrame):
    ''' '''
    
    def __init__(self, parent=None):
        super(HLine, self).__init__(parent)
        self.setFrameStyle(QtGui.QFrame.HLine)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)

class MainWindow(QtGui.QMainWindow):
    ''' '''
    
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.initUI()
        
    def initUI(self):
        '''Initialise UI'''
        
        # initial settings
        self.parent_dir_base = os.getcwd()
        self.result_dir_base = RES_DIR
        self.coef_dir_base = COEF_DIR
        
        # main window parameters
        geo = [100, 100, 400, 100]
        self.setGeometry(*geo)
        self.setWindowTitle('Testing')
        
        vbox = QtGui.QVBoxLayout()
        
        # AB File line
        
        hb1 = QtGui.QHBoxLayout()
        
        l1 = QtGui.QLabel()
        l1.setText('Sol File:')
        
        self.i1 = QtGui.QLineEdit(self)
        b1 = QtGui.QPushButton(self, text='...')
        b1.clicked.connect(self.browse_sol_file)
        
        hb1.addWidget(l1, stretch=0)
        hb1.addWidget(self.i1, stretch=5)
        hb1.addWidget(b1, stretch=0)
        
        # coef file line
        
        hb2 = QtGui.QHBoxLayout()
        
        l2 = QtGui.QLabel()
        l2.setText('Coef file:')
        
        self.i2 = QtGui.QLineEdit(self)
        b2 = QtGui.QPushButton(self, text='...')
        b2.clicked.connect(self.browse_coef_file)
        
        hb2.addWidget(l2, stretch=0)
        hb2.addWidget(self.i2, stretch=5)
        hb2.addWidget(b2, stretch=0)
        
        # Command buttons
        
        hb3 = QtGui.QHBoxLayout()

        b3 = QtGui.QPushButton(self, text='Process')
        b3.clicked.connect(self.run_processing)
        
        b4 = QtGui.QPushButton(self, text='Exit')
        b4.clicked.connect(QtCore.QCoreApplication.instance().quit)
        
        if True:
            self.i3 = QtGui.QLineEdit(self)
            self.i3.setText('MDS')
            hb3.addWidget(self.i3, stretch=3)
        else:
            hb3.addStretch(stretch=3)

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
    
    def browse_sol_file(self):
        ''' '''
        
        sol_fname = str(QtGui.QFileDialog.getOpenFileName(
            self, 'Select AB File...', self.result_dir_base))
        
        if sol_fname:
            self.result_dir_base = os.path.dirname(sol_fname)
            self.i1.setText(sol_fname)
    
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
            raise IOError
        
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
    
    def run_processing(self):
        ''' '''
        
        sol_fname = os.path.normpath(str(self.i1.text()))
        coef_fname = os.path.normpath(str(self.i2.text()))
        
         # confirm fname  format
        if not re.match('.+[.]json', sol_fname):
            print('Invalid filename format...')
            return
            
        if not coef_fname:
            print('Missing coef file')
            return
        
        try:
            sol = Solution(sol_fname)
        except IOError:
            print('Failed to read solution file')
            return

        try:
            # load coef file
            h, J = self.load_coef_file(coef_fname)
            efunc = lambda s: compute_E(h, J, s)
        except:
            print('Invalid coef file given...')
        
        # outcome statistics
        cell_occ = sol.cell_occ
        
        N = max(x[0] for x in cell_occ)
        M = max(x[1] for x in cell_occ)
        D = np.zeros([M, N], dtype=int)
        
        print(D.shape)
        
        for i, j, v in cell_occ:
            D[j-1,i-1] = v
        
        # reject rare outcomes
        if False:
            inds = np.nonzero(np.max(D, axis=0) > 2)[0]
            D = D[:, inds]
        
        if False:
            if True:
                stack_plot = stackPlot(label='GT')
                stack_plot.set_data(D)
                stack_plot.plot(block=True)
            else:
                plt.figure('GT')
                plt.clf()
                plt.plot(D.transpose(), 'x')
                plt.show(block=True)
                
        # statistical distance
        SD = np.zeros([M, M], dtype=float)
        
        print('Computing statistical distances...')
        k = 0
        for i in range(M-1):
            for j in range(i+1, M):
                k += 1
                sys.stdout.write('\r{0}%'.format(k*100./(.5*M*(M-1))))
                sys.stdout.flush()
                SD[i,j] = SD[j,i] = stat_dist(D[i,:], D[j,:])
        print('\n')
        
        # seriate SD matrix if at least one pair of GTs have SD overlap
        mask = np.ones(SD.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        if np.min(SD[mask])>0:
            print('seriating SD matrix...')
            try:
                print('\tattempting {0}'.format(self.i3.text()))
                new_inds = seriate(SD, method=str(self.i3.text()))
            except:
                print('\tfailed, using default method')
                new_inds = seriate(SD, method='MDS')
            print(new_inds)
            
            SD = SD[new_inds, :][:, new_inds]
        else:
            print('No overlap between GT distributions. Seriation not possible.')
        
        plt.imshow(SD, aspect='auto', interpolation='none', vmin=0, vmax=1.0)
        plt.colorbar()
        plt.show(block=True)
        return
        
        avg_pdf = np.sum(D,axis=0)*1./np.sum(D)
        
        # look at number of random GT needed to estimate avg_pdf
        sd_est = {k: match_dist(D, avg_pdf, k) for k in range(1, M)}
        
        pprint(sd_est)
    
        K = sorted(sd_est.keys())
        
        plt.figure('SD v K')
        plt.plot(K, [sd_est[x][0] for x in K], 'x')
        plt.xlabel('Number of samples', fontsize=FS)
        plt.ylabel('Statistical Distance', fontsize=FS)
        plt.show(block=False)
        
        plt.figure('stdev(SD) v K')
        plt.plot(K, [sd_est[x][1] for x in K])
        plt.xlabel('Number of samples', fontsize=FS)
        plt.ylabel('$\sigma_{SD}$', fontsize=FS)
        plt.show(block=True)
        
    
def main():
    '''Main loop which initialises application'''
    
    app = QtGui.QApplication(sys.argv)
    
    w = MainWindow()
    w.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()