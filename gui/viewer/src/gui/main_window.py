

from PyQt4 import QtGui, QtCore
import gui_settings as settings
from qca_widget import QCAWidget
from chimera_widget import ChimeraWidget
from core.solution import Solution

import os

class MainWindow(QtGui.QMainWindow):
    ''' '''
    
    def __init__(self, parent=None):
        ''' '''
        
        super(MainWindow, self).__init__(parent)
        
        self.initUI()
        
    def initUI(self):
        ''' '''
        
        # default parameters
        self.solution_dir = os.getcwd()

        # main window parameters
        geo = [settings.WIN_X0, settings.WIN_Y0,
               settings.WIN_DX, settings.WIN_DY]
        self.setGeometry(*geo)
        self.setWindowTitle('Result Viewer')

        self.statusBar()

        # build the menu
        self.init_menubar()

        # build the toolbar
        self.init_toolbar()
        
        # setup main layout
        hbox = QtGui.QHBoxLayout()
        
        # QCA widget
        self.qca_widget = QCAWidget(self)
        
        # Chimera widget
        self.chimera_widget = ChimeraWidget(self)
        
        hbox.addWidget(self.qca_widget, stretch=4)
        hbox.addWidget(self.chimera_widget, stretch=4)
        
        main_widget = QtGui.QWidget(self)
        main_widget.setLayout(hbox)
        self.setCentralWidget(main_widget)

    def init_menubar(self):
        ''' '''
        
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('&File')
#        tool_menu = menubar.addMenu('&Tools')
        
        
        ## create actions
        
        # loading methods
        
        solutionFileAction = QtGui.QAction(
            QtGui.QIcon(settings.ICO_DIR+'load-sol.png'),
            'Open Solution file...', self)
        solutionFileAction.triggered.connect(self.load_sol_file)
        
        # saving methods
        
        # analysis methods
        
        # SVG exporting
        
        # exit
        
        ## add actions to menus
        
        file_menu.addAction(solutionFileAction)
        
    
    def init_toolbar(self):
        ''' '''
        
        toolbar = QtGui.QToolBar()
        toolbar.setIconSize(QtCore.QSize(settings.ICO_SIZE, settings.ICO_SIZE))
        self.addToolBar(QtCore.Qt.LeftToolBarArea, toolbar)
        
        ## construct actions
        
        solutionFileAction = QtGui.QAction(self)
        solutionFileAction.setIcon(
            QtGui.QIcon(settings.ICO_DIR+'load-sol.png'))
        solutionFileAction.setStatusTip('Open Solution file...')
        solutionFileAction.triggered.connect(self.load_sol_file)
        
        ## add actions to toolbar
        toolbar.addAction(solutionFileAction)
    
    
    # ACTIONS
    
    def load_sol_file(self):
        '''Prompt user for solution file (JSON format)'''
        
        fname = str(QtGui.QFileDialog.getOpenFileName(
            self, 'Select Solution File', self.solution_dir,
            filter='JSON (*.json)'))
        
        if not fname:
            return
    
        # update solution home directory
        fdir = os.path.dirname(fname)
        self.solution_dir = fdir
        
        try:
            solution = Solution(fname)
        except Exception as e:
            print(e.message)
            print('Failed to parse solution file...')
            return

    # EVENT HANDLERS