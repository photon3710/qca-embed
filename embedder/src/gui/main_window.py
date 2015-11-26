#!/usr/bin/env python

# -----------------------------------
# Name: application.py
# Desc: Main loop for QCA embedder application
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

        # QCA widget
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
        return

        menubar = self.menuBar()

        file_menu = menubar.addMenu('&File')
        tool_menu = menubar.addMenu('&Tools')
        view_menu = menubar.addMenu('&View')
        help_menu = menubar.addeMenu('&Help')

        # construct actions

    def init_toolbar(self):
        ''' '''

        toolbar = QtGui.QToolBar()
        toolbar.setIconSize(QtCore.QSize())

    def test(self):
        ''' '''
        pass
