#!/usr/bin/env python

from PyQt4 import QtGui
from gui.mw_viewer import MainWindow
import sys


def main():
    '''Main loop which initialises embedder application'''

    app = QtGui.QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

