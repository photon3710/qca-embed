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
from gui.main_window import MainWindow
import sys


def main():
    '''Main loop which initialises embedder application'''

    app = QtGui.QApplication(sys.argv)

    w = MainWindow()
    w.show()

    app.exec_()


if __name__ == '__main__':
    main()
