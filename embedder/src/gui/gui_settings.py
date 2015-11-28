#!/usr/bin/env python

# -----------------------------------
# Name: application.py
# Desc: Main loop for QCA embedder application
# Author: Jake Retallick
# Created: 2015.11.25
# Modified: 2015.11.25
# Licence: Copyright 2015
# -----------------------------------

from PyQt4 import QtGui

# MAIN WINDOW SETTINGS
WIN_DX = 1200   # width of the main window
WIN_DY = 600    # height of the main window
WIN_X0 = 100    # x-offset of the main window
WIN_Y0 = 100    # y-offset of the main window

ICO_SIZE = 30           # icon size
ICO_DIR = './gui/ico/'   # icon directory

BUTTON_SIZE = 25    # size of buttons

# QCA CELL PARAMETERS
CELL_SEP = 70               #
CELL_SIZE = .8*CELL_SEP     #
CELL_ALPHA = int(.15*255)   #

# --colors
QCA_COL = {'default': QtGui.QColor(255, 255, 255),
           'inactive': QtGui.QColor(100, 100, 100),
           'output': QtGui.QColor(0, 200, 0, 150),
           'input': QtGui.QColor(200, 0, 0, 150),
           'fixed': QtGui.QColor(255, 165, 0, 150)}

DOT_RAD = 0.25*CELL_SIZE

# --qca pen
CELL_PEN_WIDTH = max(1, int(0.05*CELL_SIZE))    #
CELL_PEN_COLOR = QtGui.QColor(180, 180, 180)    #
TEXT_PEN_WIDTH = max(1, int(0.05*CELL_SIZE))    #
TEXT_PEN_COLOR = QtGui.QColor(0, 0, 0)          #

# --qca magnification
MAX_MAG = 5             # maximum magnification
MIN_MAG = 0.1           # minimum magnification
MAG_STEP = 0.1          # magnification step
MAG_WHEEL_FACT = 0.2    # factor for wheel zoom\

QCA_CANVAS_OFFSET = 0.2
