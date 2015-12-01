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

from pprint import pprint


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
        self.use_dense = True       # True if using Dense Placement embedder

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
        tool_menu = menubar.addMenu('&Tools')
#        view_menu = menubar.addMenu('&View')
#        help_menu = menubar.addMenu('&Help')

        # construct actions

        qcaFileAction = QtGui.QAction(
            QtGui.QIcon(settings.ICO_DIR+'qca_file.png'),
            'Open QCA file...', self)
        qcaFileAction.triggered.connect(self.load_qca_file)

        embedFileAction = QtGui.QAction(
            QtGui.QIcon(settings.ICO_DIR+'open_embed.png'),
            'Open EMBED file...', self)
        embedFileAction.triggered.connect(self.load_embed_file)

        chimeraFileAction = QtGui.QAction(
            QtGui.QIcon(settings.ICO_DIR+'chimera_file.png'),
            'Open chimera file...', self)
        chimeraFileAction.triggered.connect(self.load_chimera_file)

        exitAction = QtGui.QAction('Exit', self)
        exitAction.setShortcut('Ctrl+W')
        exitAction.triggered.connect(self.close)
        
        self.action_dense_embed_flag = QtGui.QAction('Dense', self)
        self.action_dense_embed_flag.triggered.connect(self.switch_embedder)
        self.action_dense_embed_flag.setEnabled(False)
        
        self.action_heur_embed_flag = QtGui.QAction('Heuristic', self)
        self.action_heur_embed_flag.triggered.connect(self.switch_embedder)
        self.action_heur_embed_flag.setEnabled(True)

        file_menu.addAction(qcaFileAction)
#        file_menu.addAction(embedFileAction)
        file_menu.addAction(chimeraFileAction)
        file_menu.addSeparator()
        file_menu.addAction(exitAction)
        
        embedder_menu = tool_menu.addMenu('Embedding method')
        embedder_menu.addAction(self.action_dense_embed_flag)
        embedder_menu.addAction(self.action_heur_embed_flag)

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
        
        self.action_embed = QtGui.QAction(self)
        self.action_embed.setIcon(
            QtGui.QIcon(settings.ICO_DIR+'embed.png'))
        self.action_embed.setStatusTip('Embed diplayed circuit...')
        self.action_embed.triggered.connect(self.embed_circuit)
        self.action_embed.setEnabled(False)

        toolbar.addAction(action_qca_file)
#        toolbar.addAction(action_embed_file)
        toolbar.addAction(action_chimera_file)
        toolbar.addAction(self.action_switch_adj)
        toolbar.addAction(self.action_embed)

    def load_qca_file(self):
        '''Prompt filename for qca file'''

        fname = QtGui.QFileDialog.getOpenFileName(
            self, 'Select QCA File', self.qca_dir)

        # update qca home directory
        fdir = os.path.dirname(fname)
        self.qca_dir = fdir

        adj_typ = 'full' if self.full_adj else 'lim'
        self.qca_widget.updateCircuit(fname, adj_typ=adj_typ)
        
        if not self.qca_active:
            self.qca_active = True
            self.action_embed.setEnabled(True)
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
            sub_message = 'Limited' if self.full_adj else 'Full'
            self.action_switch_adj.setIcon(
                QtGui.QIcon(settings.ICO_DIR+ico_file))
            self.action_switch_adj.setStatusTip(
                'Switch to {0} Adjacency...'.format(sub_message))
            self.qca_widget.setAdjacency('full' if self.full_adj else 'lim')
        
    def switch_embedder(self):
        '''Change between embedding algorithms and set menu enabling'''
        
        self.action_dense_embed_flag.setEnabled(self.use_dense)
        self.action_heur_embed_flag.setEnabled(not self.use_dense)
        self.use_dense = not self.use_dense
        
    def embed_circuit(self):
        '''Run embedding on displayed circuit into selected chimera 
        sub-graph'''
        
        print('Running embedding...')
        
        # get circuit configuration
        
        # get chimera sub-graph
        M, N, adj, active_range = self.chimera_widget.getActiveGraph()
        
        pprint(adj)

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
