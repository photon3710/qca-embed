#!/usr/bin/env python

from PyQt4 import QtGui, QtCore

import sys, os, re, json
import numpy as np
import matplotlib.pyplot as plt

# MAIN WINDOW SETTINGS
WIN_DX = 1400   # width of the main window
WIN_DY = 800    # height of the main window
WIN_X0 = 100    # x-offset of the main window
WIN_Y0 = 100    # y-offset of the main window

class MainWindow(QtGui.QMainWindow):
    ''' '''

    def __init__(self, parent=None):
        '''Initialise Viewer main window '''
        super(MainWindow, self).__init__(parent)
        self.initUI()

    def initUI(self):
        '''Initialise the user interface'''

        # directory parameters
        self.qca_dir = os.getcwd()      # look-up dir for QCAD files
        self.sol_dir = os.getcwd()      # look-up dir for D-Wave results
        self.embed_dir = os.getcwd()    # look-up dir for embed and coef files

        self.chimera_dir = os.getcwd()  # look-up dir for chimera file
        self.svg_dir = os.getcwd()      # look-up dir for SVG saving

        # functional parameters
        self.chimera_file = None    # if using actual

        # main window parameters
        geo = [WIN_X0, WIN_Y0, WIN_DX, WIN_DY]
        self.setGeometry(*geo)
        self.setWindowTitle('Result Viewer')

        self.statusBar()

        # build the menubar
        self.init_menubar()

        # build the toolbar
        self.init_toolbar()

    def init_menubar(self):
        '''Set up menubar'''
        pass

    def init_toolbar(self):
        '''Set up toolbar'''
        pass
