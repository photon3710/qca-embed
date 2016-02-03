from PyQt4 import QtGui, QtCore

from pprint import pprint

import sys
import os
import re


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
    
        vbox.addLayout(hb1)    
        vbox.addWidget(HLine())
        vbox.addLayout(vb1)
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

    def load_embed_file(self, fname):
        ''' '''
        
        try:
            fp = open(fname, 'r')
        except IOError:
            print('Failed to load embed file: {0}'.format(os.path.basename(fname)))
            return
            
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
        
        full_adj = all_data[1]['full_data'] == 'True'
        use_dense = all_data[1]['use_dense'] == 'True'
        
        M, N, L, M0, N0 = [int(all_data[2][k]) for k in ['M', 'N', 'L', 'M0', 'N0']]
        regex = re.compile('[0-9]+')
        fmap = lambda qb: [int(x) for x in regex.findall(qb)]
        models = {int(k): [fmap(qb) for qb in all_data[3][k].split(';')]  for k in all_data[3]}
                    

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
            i, fn = line.strip().split(':')
            data[int(i)] = os.join(dir_name, fn.strip())

        fp.close()
        
        # load all 
        return data

    def run_processing(self):
        ''' '''
        
        parent_dir = os.path.normpath(str(self.i1.text()))
        
        # get .info and .embed file from parent_dir
        try:
            pol_fn = parent_dir+'/coefs/pols.info'
            summ_fn = parent_dir+'/embed/summary.embed'
            # process pol and summary file
            pol_data = self.process_pol_file(pol_fn)
            embed_data = self.process_summ_file(summ_fn)
        except IOError:
            print('Invalid file architecture...')
            return

        sol_files = str(self.i2.toPlainText()).split('\n')
        
        
        
    
def main():
    '''Main loop which initialises application'''

    app = QtGui.QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()