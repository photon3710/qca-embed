#!/usr/bin/env python

# -----------------------------------
# Name: application.py
# Desc: Main loop for D-Wave solution viewer application
# Author: Jake Retallick
# Created: 2015.12.15
# Modified: 2015.12.15
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

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
