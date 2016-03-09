from PyQt4 import QtGui, QtCore
from temp.qca_widget import QCAWidget
from temp.chimera_widget import ChimeraWidget

import sys
import os
import re
import json
import numpy as np

from pprint import pprint

RESULT_DIR = os.path.abspath(os.path.join(os.pardir, 'results'))

# MAIN WINDOW SETTINGS
WIN_DX = 1000   # width of the main window
WIN_DY = 600    # height of the main window
WIN_X0 = 100    # x-offset of the main window
WIN_Y0 = 100    # y-offset of the main window

ICO_SIZE = 50           # icon size
ICO_DIR = './gui/ico'   # icon directory

BUTTON_SIZE = 25        # size of buttons

## GENERAL FUNCTIONS

def get_directory_architecture(root):
    '''Construct a nested dictionary describing the directory structure within
    the given root directory'''
    
    if not os.path.isdir(root):
        print('Not a valid directory...')
        return None
    arch = {}
    root = os.path.normpath(root)
    start_ind = root.rfind(os.sep)+1
    for path, dirs, files in os.walk(root):
        dirs = path[start_ind:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, dirs[:-1], arch)
        parent[dirs[-1]] = subdir
    return arch

## MAIN WINDOW CLASS

class MainWindow(QtGui.QMainWindow):
    ''' '''
    
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()
    
    def initUI(self):
        ''' '''
        
        # default parameters
        self.pind = 0       # current polarization index
        self.pinds = ['0']  # list of polarization labels
        self.pind_c = 1     # number of polarization indices
        self.result_dir = RESULT_DIR    # result directory

        # persistent parameters
        self.full_adj = True    # persistent adjacency type
        self.rt = 20            # persistent annealing time
        self.qca_file = None    # path from result_dir to qca_file
        self.result_par = {'cat': '',
                           'circ': '',
                           'adj': '',
                           'ind': '',
                           'rt': '',
                           'pind': ''}
        self.data = {}          # result file data from json

        self.cat_menus = {}     # category menus
        self.circ_actions = {}  # circuit sub-actions
    
        # build GUI
        main_geo = [WIN_X0, WIN_Y0, WIN_DX, WIN_DY]
        self.setGeometry(*main_geo)
        self.setWindowTitle('Solution Viewer')

        self.statusBar()
        self.init_menubar()
        self.init_toolbar()
        
        # Set up results menu
        self.build_results_directory()  
        
        self.qca_widget = QCAWidget(parent=self)
        self.setCentralWidget(self.qca_widget)
#        self.chimera_widget = ChimeraWidget(parent=self)
#        self.display_widget = QtGui.QWidget(parent=self)
#        
#        left_vb = QtGui.QVBoxLayout()
#        right_vb = QtGui.QVBoxLayout()
#        main_layout = QtGui.QHBoxLayout()
#        
#        left_vb.addWidget(self.qca_widget, stretch=1)
#        left_vb.addWidget(self.chimera_widget, stretch=1)
#        
#        right_vb.addWidget(self.display_widget, stretch=1)
#        main_layout.addLayout(left_vb, stretch=1)
#        main_layout.addLayout(right_vb, stretch=1)
#        
#        central_widget = QtGui.QWidget(self)
#        central_widget.setLayout(main_layout)
#        self.setCentralWidget(central_widget)
    
    def init_menubar(self):
        '''Setup menus'''
        
        menu = self.menuBar()

        # static menus
        file_menu = menu.addMenu('&File')
        tool_menu = menu.addMenu('&Tool')
        
        # active menus
        self.result_menu = menu.addMenu('&Results')
        
    def init_toolbar(self):
        '''Setup toolbar buttons'''
        pass
    
    # actions
    
    def update_widgets(self, new_qca=False):
        ''' '''
        
        if new_qca:
            qca_name = os.path.normpath(os.path.join(self.result_dir, self.data['qca_file']))
            # change qca_widget
            self.qca_widget.update_circuit(qca_name, self.full_adj)

    def build_results_directory(self):
        ''' '''
        
        self.arch = get_directory_architecture(self.result_dir)[os.path.basename(self.result_dir)]
        if not self.arch:
            print('Invalid result directory...')

        if self.cat_menus or self.circ_actions:
            # clear result menu
            self.result_menu.clear()
            # clear stored menus
            self.cat_menus = {}
            self.circ_actions = {}
        
        # build new stored menus
        for cat in self.arch:
            circs = self.arch[cat].keys()
            self.cat_menus[cat] = self.result_menu.addMenu(cat)
            self.circ_actions[cat] = {}
            for circ in circs:
                action = QtGui.QAction(circ, self)
                func = lambda check, cat=cat, circ=circ: self.switch_circuit(cat, circ)
                action.triggered.connect(func)
                self.cat_menus[cat].addAction(action)
                self.circ_actions[cat][circ] = action
        
    def switch_circuit(self, cat, circ):
        ''' '''
        
        # get sub-architecture if available
        try:
            sub_arch = self.arch[cat][circ]
        except KeyError:
            print('Key conflict in circuit selection... somehow')
            return

        ## attempt to switch to most similar case
        
        # adjacency type: assume either 'full' or 'lim' adjacency
        adj_map = lambda full_adj: 'full' if full_adj else 'lim'
        if adj_map(self.full_adj) not in sub_arch:
            print('Switching adjacency type...')
            self.full_adj = not self.full_adj
        adj = adj_map(self.full_adj)
        
        # pick lowest circuit index
        ind = str(min([int(x) for x in sub_arch[adj]]))
        
        # annealing time, if not available use 20 else lowest
        if str(self.rt) not in sub_arch[adj][ind]:
            if '20' in sub_arch[adj][ind]:
                self.rt = 20
            else:
                self.rt = min([int(x) for x in sub_arch[adj][ind]])
        rt = str(self.rt)
        
        # get list of all available pinds
        regex = re.compile('^sol[0-9]+.json$')
        pinds = filter(regex.match, sub_arch[adj][ind][rt])
        try:
            pinds = sorted([int(re.search('[0-9]+', x).group(0)) for x in pinds])
        except AttributeError:
            print('Invalid result filename...')
            return
        
        # can't assume relationship between pinds so set pinds=0
        self.pinds = [str(x) for x in pinds]
        self.pind = 0
        self.pind_max = len(pinds)
        
        # set persistent result_par dict
        self.result_par['cat'] = cat
        self.result_par['circ'] = circ
        self.result_par['adj'] = adj
        self.result_par['ind'] = int(ind)
        self.result_par['rt'] = int(self.rt)
        self.result_par['pind'] = int(self.pind)
        
        self.load_result()
    
    def process_result(self, fname):
        '''Process the json file for simplified use'''
        
        # correct path to qca_file
        self.data['qca_file'] = os.path.join(fname, self.data['qca_file'])
        

    def load_result(self):
        ''' '''

        # get filename string
        rp = self.result_par
        fname = os.path.join(rp['cat'], rp['circ'], rp['adj'], str(rp['ind']),
                             str(rp['rt']), 'sol{0}.json'.format(self.pinds[self.pind]))
        fname = os.path.join(self.result_dir, fname)

        # attempt to load json file
        try:
            fp = open(fname, 'r')
        except IOError:
            print('Failed to load json file: {0}'.format(fname))
            raise IOError

        self.data = json.load(fp)
        fp.close()
        
        # process parameters
        self.process_result(fname)
        
        # update widgets
        self.update_widgets(new_qca=(self.qca_file != self.data['qca_file']))
        
        self.qca_file = self.data['qca_file']
        
    
    def incdec_pind(self, inc=True):
        '''Increment or decrement self.pind'''
        
        # exit if only one possible pind
        if self.pind_max==1:
            return
        
        self.pind = (self.pind + (1 if inc else -1))%self.pind_max
        self.result_par['pind'] = self.pind
        self.load_result()
        
    def incdec_rt(self, inc=True):
        '''Increase or Decrease annealing time'''
        pass
    
    def incdec_ind(self, inc=True):
        '''Increment or decrement circuit index'''
        pass
        
    # interrupts
    
    def keyPressEvent(self, e):
        ''' '''
        
        mods = QtGui.QApplication.keyboardModifiers()   # binary-like flag
        shift = (mods == QtCore.Qt.ShiftModifier)

        if e.key() == QtCore.Qt.Key_P:
            # inc/dec polarization index if possible
            self.incdec_pind(inc=(not shift))
        elif e.key() == QtCore.Qt.Key_T:
            # inc/dec runtime
            self.incdec_rt(inc=(not shift))
        elif e.key() == QtCore.Qt.Key_I:
            # inc/dec circuit index
            self.incdec_ind(inc=(not shift))
            
def main():
    '''Main loop which initialises application'''

    app = QtGui.QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
