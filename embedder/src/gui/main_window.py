#!/usr/bin/env python

# -----------------------------------
# Name: maiMain Window widget for embedder application
# Author: Jake Retallick
# Created: 2015.11.25
# Modified: 2015.11.25
# Licence: Copyright 2015
# -----------------------------------

from PyQt4 import QtGui, QtCore
import os

import gui_settings as settings
from qca_widget import QCAWidget
from chimera_widget import ChimeraWidget


class MainWindow(QtGui.QMainWindow):
    '''Main Window widget for embedder application'''

    def __init__(self):
        '''Create the Main Window widget'''

        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        '''Initialise the UI'''

        # default parameters
        self.qca_dir = os.getcwd()
        self.embed_dir = os.getcwd()
        self.chimera_dir = os.getcwd()
        
        # functionality paremeters
        self.qca_active = False     # True when QCAWidget set
        self.full_adj = True        # True when using full adjacency

        # main window parameters
        geo = [settings.WIN_X0, settings.WIN_Y0,
               settings.WIN_DX, settings.WIN_DY]
        self.setGeometry(*geo)
        self.setWindowTitle('QCA Embedder')

        self.statusBar()

        # build the menu
        self.init_menubar()

        # build the toolbar
        self.init_toolbar()

        # set up the main layout
        hbox = QtGui.QHBoxLayout()

        # QCA widget placeholder
        self.qca_widget = QCAWidget()

        # Chimera widget
        self.chimera_widget = ChimeraWidget()

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
#        view_menu = menubar.addMenu('&View')
#        help_menu = menubar.addMenu('&Help')

        # construct actions

        qcaFileAction = QtGui.QAction(
            QtGui.QIcon(settings.ICO_DIR+'qca.png'),
            'Open QCA file...', self)
        qcaFileAction.triggered.connect(self.load_qca_file)

        embedFileAction = QtGui.QAction(
            QtGui.QIcon(settings.ICO_DIR+'open_embed.png'),
            'Open EMBED file...', self)
        embedFileAction.triggered.connect(self.load_embed_file)

        chimeraFileAction = QtGui.QAction(
            QtGui.QIcon(settings.ICO_DIR+'chimera.png'),
            'Open chimera file...', self)
        chimeraFileAction.triggered.connect(self.load_chimera_file)

        exitAction = QtGui.QAction('Exit', self)
        exitAction.setShortcut('Ctrl+W')
        exitAction.triggered.connect(self.close)

        file_menu.addAction(qcaFileAction)
#        file_menu.addAction(embedFileAction)
        file_menu.addAction(chimeraFileAction)
        file_menu.addSeparator()
        file_menu.addAction(exitAction)

    def init_toolbar(self):
        ''' '''

        toolbar = QtGui.QToolBar()
        toolbar.setIconSize(QtCore.QSize(settings.ICO_SIZE, settings.ICO_SIZE))
        self.addToolBar(QtCore.Qt.LeftToolBarArea, toolbar)

        # construct actions
        action_qca_file = QtGui.QAction(self)
        action_qca_file.setIcon(
            QtGui.QIcon(settings.ICO_DIR+'qca_file.png'))
        action_qca_file.setStatusTip('Open QCA file...')
        action_qca_file.triggered.connect(self.load_qca_file)

        action_embed_file = QtGui.QAction(self)
        action_embed_file.setIcon(
            QtGui.QIcon(settings.ICO_DIR+'embed_file.png'))
        action_embed_file.setStatusTip('Open embedding file...')
        action_embed_file.triggered.connect(self.load_embed_file)

        action_chimera_file = QtGui.QAction(self)
        action_chimera_file.setIcon(
            QtGui.QIcon(settings.ICO_DIR+'chimera_file.png'))
        action_chimera_file.setStatusTip('Open chimera file...')
        action_chimera_file.triggered.connect(self.load_chimera_file)
        
        self.action_switch_adj = QtGui.QAction(self)
        self.action_switch_adj.setIcon(
            QtGui.QIcon(settings.ICO_DIR+'lim_adj.png'))
        self.action_switch_adj.setStatusTip('Switch to Limited Adjacency...')
        self.action_switch_adj.triggered.connect(self.switch_adjacency)
        self.action_switch_adj.setEnabled(False)

        toolbar.addAction(action_qca_file)
#        toolbar.addAction(action_embed_file)
        toolbar.addAction(action_chimera_file)
        toolbar.addAction(self.action_switch_adj)

    def load_qca_file(self):
        '''Prompt filename for qca file'''

        fname = QtGui.QFileDialog.getOpenFileName(
            self, 'Select QCA File', self.qca_dir)

        # update qca home directory
        fdir = os.path.dirname(fname)
        self.qca_dir = fdir

        self.qca_widget.updateCircuit(fname)
        
        if not self.qca_active:
            self.qca_active = True
            self.action_switch_adj.setEnabled(True)

    def load_embed_file(self):
        '''Prompt filename for embed file'''

        fname = QtGui.QFileDialog.getOpenFileName(
            self, 'Select Embedding File', self.embed_dir)

        # update embed home directory
        fdir = os.path.dirname(fname)
        self.embed_dir = fdir

        # do stuff

    def load_chimera_file(self):
        '''Prompt filename for chimera structure'''

        fname = QtGui.QFileDialog.getOpenFileName(
            self, 'Select Chimera File', self.chimera_dir)

        # update chimera home directory
        fdir = os.path.dirname(fname)
        self.chimera_dir = fdir

        self.chimera_widget.updateChimera(fname)
        
    def switch_adjacency(self):
        ''' '''
        
        if self.qca_active:
            self.full_adj = not self.full_adj
            ico_file = 'lim_adj.png' if self.full_adj else 'full_adj.png'
            self.action_switch_adj.setIcon(
                QtGui.QIcon(settings.ICO_DIR+ico_file))
            self.qca_widget.setAdjacency('full' if self.full_adj else 'lim')

    def closeEvent(self, e):
        '''Handle main window close event'''

        reply = QtGui.QMessageBox.question(
            self, 'Message', 'Are you sure you want to quit?',
            QtGui.QMessageBox.Yes | QtGui.QMessageBox.Cancel,
            QtGui.QMessageBox.Cancel)

        if reply == QtGui.QMessageBox.Yes:
            e.accept()
        else:
            e.ignore()
