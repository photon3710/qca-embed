from PyQt4 import QtGui, QtCore

from core.dwave_sol import DWAVE_Sol
from core.chimera import tuple_to_linear

import sys
import os
import re
import json
import numpy as np

from collections import defaultdict
from pprint import pprint

SOL_DIR = 'results/DWAVE'
SOL_DIR = os.path.join(os.getcwd(), os.pardir, SOL_DIR)

def fprint(s):
    print(s)
    sys.stdout.flush()

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

    def __init__(self):
        ''' '''

        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        ''' '''

        # initial settings
        self.parent_dir_base = os.getcwd()
        self.result_dir_base = SOL_DIR
        self.solution_dir_base = os.getcwd()

        # main window parameters
        geo = [100, 100, 600, 400]
        self.setGeometry(*geo)
        self.setWindowTitle('Solution Processing')

        vbox = QtGui.QVBoxLayout()

        # parent directory line
        hb1 = QtGui.QHBoxLayout()

        l1 = QtGui.QLabel()
        l1.setText('Parent Dir:')

        self.i1 = QtGui.QLineEdit(self)
        b1 = QtGui.QPushButton(self, text='...')
        b1.clicked.connect(self.browse_parent_dir)

        hb1.addWidget(l1, stretch=1)
        hb1.addWidget(self.i1, stretch=3)
        hb1.addWidget(b1, stretch=0)

        # second section

        vb1 = QtGui.QVBoxLayout()

        l2 = QtGui.QLabel()
        l2.setText('Solution Files:')

        self.i2 = QtGui.QTextEdit(self)

        hb2 = QtGui.QHBoxLayout()
        self.i3 = QtGui.QLineEdit(self)
        self.i3.returnPressed.connect(self.enter_solution)

        b2 = QtGui.QPushButton(self, text='...')
        b2.clicked.connect(self.browse_solution_files)

        hb2.addWidget(self.i3, stretch=4)
        hb2.addWidget(b2, stretch=0)

        vb1.addWidget(l2, stretch=0)
        vb1.addWidget(self.i2, stretch=3)
        vb1.addLayout(hb2, stretch=0)

        # third line
        hb3 = QtGui.QHBoxLayout()

        b3 = QtGui.QPushButton(self, text='Process')
        b3.clicked.connect(self.run_processing)

        b4 = QtGui.QPushButton(self, text='Exit')
        b4.clicked.connect(QtCore.QCoreApplication.instance().quit)

        hb3.addWidget(b3)
        hb3.addWidget(b4)

        # fourth line

        hb4 = QtGui.QHBoxLayout()

        l3 = QtGui.QLabel()
        l3.setText('Output Directory:')

        self.i4 = QtGui.QLineEdit(self)
        self.i4.setText(SOL_DIR)

        b3 = QtGui.QPushButton(self, text='...')
        b3.clicked.connect(self.browse_result_dir)

        hb4.addWidget(l3, stretch=0)
        hb4.addWidget(self.i4, stretch=3)
        hb4.addWidget(b3, stretch=0)

        vbox.addLayout(hb1)
        vbox.addWidget(HLine())
        vbox.addLayout(vb1)
        vbox.addWidget(HLine())
        vbox.addLayout(hb4)
        vbox.addWidget(HLine())
        vbox.addLayout(hb3)

        # central widget
        cw = QtGui.QWidget(self)
        cw.setLayout(vbox)
        self.setCentralWidget(cw)

    def browse_parent_dir(self):
        ''' '''

        dir_name = str(QtGui.QFileDialog.getExistingDirectory(
            self, 'Select parent directory...', self.parent_dir_base))

        if dir_name:
            self.parent_dir_base = dir_name
            self.i1.setText(dir_name)

    def browse_result_dir(self):
        ''' '''
        dir_name = str(QtGui.QFileDialog.getExistingDirectory(
            self, 'Select result directory...', self.result_dir_base))

        if dir_name:
            self.result_dir_base = dir_name
            self.i4.setText(dir_name)

    def browse_solution_files(self):
        ''' '''

        sol_files = QtGui.QFileDialog.getOpenFileNames(
            self, 'Select solution files...', self.solution_dir_base)

        sol_files = [str(s) for s in sol_files]
        if sol_files:
            self.solution_dir_base = os.path.dirname(sol_files[0])
            for sol_file in sol_files:
                self.i2.append(sol_file)

    def enter_solution(self):
        ''' '''

        s = str(self.i3.text())
        if s:
            self.i2.append(s)
            self.i3.clear()

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
        J = defaultdict(dict)
        for line in fp:
            a, b, v = line.split()
            a, b, v = int(a), int(b), float(v)
            if a==b:
                h[a] = v
            else:
                a, b = sorted([a, b])
                J[a][b] = v
        fp.close()

        return h, J

    def load_embed_file(self, fname):
        ''' '''

        try:
            fp = open(fname, 'r')
        except IOError:
            print('Failed to load embed file: {0}'.format(os.path.basename(fname)))
            raise IOError

        dir_name = os.path.dirname(fname)

        all_data = [{}]
        data = all_data[-1]

        for line in fp:
            if len(line)<3:
                if data:
                    all_data.append({})
                    data = all_data[-1]
            else:
                key, s = line.split(':')
                data[key] = s.strip()
        if not all_data[-1]:
            all_data.pop()

        fp.close()

        chim_file = os.path.join(dir_name, all_data[0]['chimera_file'])
        qca_file = os.path.join(dir_name, all_data[0]['qca_file'])

        full_adj = all_data[1]['full_adj'] == 'True'
        use_dense = all_data[1]['use_dense'] == 'True'

        M, N, L, M0, N0 = [int(all_data[2][k]) for k in ['M', 'N', 'L', 'M0', 'N0']]
        regex = re.compile('[0-9]+')
        fmap = lambda qb: [int(x) for x in regex.findall(qb)]
        models = {int(k): [fmap(qb) for qb in all_data[3][k].split(';')]  for k in all_data[3]}

        # append M0, N0 to models
        for k in models:
            for qb in models[k]:
                qb[0] += M0
                qb[1] += N0

        # format output to dict
        output = {'chim_file':  chim_file,
                  'qca_file':   qca_file,
                  'full_adj':   full_adj,
                  'use_dense':  use_dense,
                  'models':     models}

        return output

    def process_pol_file(self, fname):
        ''' '''

        fp = open(fname, 'r')

        all_data = [{}]
        data = all_data[0]

        # get polarization data
        regex = re.compile('[-]?[0-9]+')
        for line in fp:
            if len(line)<3:
                if data:
                    all_data.append({})
                    data = all_data[-1]
            else:
                i, pol = line.strip().split(':')
                data[int(i)] = [int(x) for x in regex.findall(pol)]
        if not all_data[-1]:
            all_data.pop()

        fp.close()

        return all_data

    def process_summ_file(self, fname):
        ''' '''

        fp = open(fname, 'r')

        chim_file = fp.readline().strip().split()[-1]
        chim_file = os.path.join(fname, chim_file)

        dir_name = os.path.dirname(fname)
        data = {}
        for line in fp:
            if len(line)<3:
                continue
            i, fn = line.strip().split(':')
            data[int(i)] = os.path.join(dir_name, fn.strip())

        fp.close()

        # load all embed files
        embeds = {}
        for k in data:
            try:
                embed = self.load_embed_file(data[k])
                embeds[k] = embed
            except IOError:
                continue

        return embeds

    def process_coef_dir(self, coef_dir):
        ''' '''

        # identify all contained coeficient files
        regex = re.compile("coefs([0-9]+)\.txt")
        coef_data = {}
        for x in os.listdir(coef_dir):
            print(x)
            if regex.match(x):
                fn = os.path.join(coef_dir, x)
                h, J = self.load_coef_file(fn)
                coef_data[int(regex.findall(x)[0])] = {'h': h, 'J': J}

        pol_fn = os.path.join(coef_dir, 'pols.info')
        pol_data = self.process_pol_file(pol_fn)

        return coef_data, pol_data

    def get_ind(self, fname):
        ''' '''

        result_dir = str(self.i4.text())
        dir_name = os.path.join(result_dir, fname)

        if not os.path.isdir(dir_name):
            return 0
        else:
            dirs = os.listdir(dir_name)
            regex = re.compile('^[0-9]+$')
            inds = [int(d) for d in dirs if regex.match(d)]
            return max(inds)+1

    def write_coef_file(self, fn, h, J, qbits):
        ''' '''

        try:
            fp = open(fn, 'w')
        except IOError:
            print('Failed to open file: {0}...'.format(fn))
            return

        fp.write('{0}\n'.format(1152))

        # write h parameters
        for k in sorted(h):
            if k in qbits:
                fp.write('{0} {1} {2:.3f}\n'.format(k, k, h[k]))

        # write J parameters
        for i in sorted(J):
            for j in sorted(J[i]):
                if i in qbits and j in qbits:
                    v = J[i][j]
                    i,j = sorted([i,j])
                    fp.write('{0} {1} {2:.3f}\n'.format(i, j, v))

        fp.close()

    def save_subsol(self, embed, pols, sol, h, J, rt, pind, sol_name):
        ''' '''

        # output file name
        result_dir = str(self.i4.text())

        base_dir = os.path.basename(os.path.dirname(embed['qca_file']))
        qca_name = os.path.splitext(os.path.basename(embed['qca_file']))[0]
        fname = os.path.join(base_dir, qca_name)
        fname = os.path.join(fname, 'full' if embed['full_adj'] else 'lim')

        # determine index
        if 'ind' in embed:
            ind = embed['ind']
        else:
            ind = self.get_ind(fname)
            embed['ind'] = ind

        fname = os.path.join(fname, str(ind))
        fname = os.path.join(fname, str(rt))
        fname = os.path.join(fname, 'sol{0}.json'.format(pind))

        fname = os.path.join(result_dir, fname)
        coef_fname = os.path.splitext(fname)[0]+'.txt'

        # build data dictionary
        data = {}
        data['qca_file'] = os.path.basename(embed['qca_file'])
        data['full_adj'] = embed['full_adj']
        data['use_dense'] = embed['use_dense']
        data['pols'] = pols
        data['models'] = embed['models']
        data['spins'] = sol.spins.tolist()
        data['occ'] = sol.occ
        data['energies'] = sol.energies
        data['sol_name'] = sol_name
        data['qbits'] = sol.qbits

        # build directory architecture
        dir_name = os.path.dirname(fname)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # output to file
        fp = open(fname, 'w')
        json.dump(data, fp, sort_keys=True)
        fp.close()

        self.write_coef_file(coef_fname, h, J, sol.qbits)

    def process_solution(self, fname, coef_data, pol_data, embeddings):
        '''Process a single solution file. It is assumed that the solution
        corresponds to the configurations set by the pol_data and embeddings.
        The filename should be of format [name][pind]_[rt]us.json, where pind
        and rt are integers. If there is only one element of pol_data there is
        no pind.'''

        fprint(os.path.basename(fname))

        NP = len(pol_data)  # number of possible polarization comfigurations
        if NP == 1:
            fn_regex = re.compile('.*[a-zA-Z_]+_[0-9]+us.json')
            val_regex = re.compile('_[0-9]+us')
        else:
            fn_regex = re.compile('.*[0-{0}]+_[0-9]+us.json'.format(NP))
            val_regex = re.compile('[0-{0}]+_[0-9]+us'.format(NP))
        if not fn_regex.match(fname):
            print('Given filename does not match the required pattern: {0}...'.format(fname))
            return

        # extract pind and rt
        val_str = val_regex.findall(fname)[-1]  # will work if this far
        vals = [int(x) for x in re.findall('[0-9]+', val_str)]

        pind = 0 if NP==1 else vals[0]
        rt = vals[-1]

#        print('{0}: pind={1}, rt={2}'.format(fname, pind, rt))

        # get solution object
        try:
            solution = DWAVE_Sol(fname)
            sol_name = os.path.basename(fname)
        except IOError:
            return

        h = coef_data[pind]['h']
        J = coef_data[pind]['J']

        efunc = lambda s: compute_E(h, J, s)

        for key in embeddings:
            embed = embeddings[key]
            pols = pol_data[pind][key]
            qbits = list(reduce(lambda x,y:x+y, embed['models'].values()))
            qbits = [tuple_to_linear(qb, M=12, N=12, L=4, index0=False) for qb in qbits]
            sol = solution.get_reduced_solution(qbits, efunc)
            self.save_subsol(embed, pols, sol, h, J, rt, pind, sol_name)

    def run_processing(self):
        ''' '''

        parent_dir = os.path.normpath(str(self.i1.text()))

        # get .info and .embed file from parent_dir
        try:
            coef_dir = os.path.join(parent_dir, 'coefs')
            summ_fn = os.path.join(parent_dir,'embed', 'summary.embed')
            # process pol and summary file
            embeddings = self.process_summ_file(summ_fn)
        except IOError:
            print('Invalid file architecture...')
            return

        coef_data, pol_data = self.process_coef_dir(coef_dir)
        sol_files = str(self.i2.toPlainText()).split('\n')

        for sol_file in sol_files:
            self.process_solution(sol_file, coef_data, pol_data, embeddings)

def main():
    '''Main loop which initialises application'''

    app = QtGui.QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
