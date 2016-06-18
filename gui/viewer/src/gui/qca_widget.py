#!/usr/bin/env python

# -----------------------------------
# Name: qca_widget.py
# Desc: Viewer widget for QCA circuits
# Author: Jake Retallick
# Created: 2015.12.15
# Modified: 2015.12.15
# Licence: Copyright 2015
# -----------------------------------

from PyQt4 import QtGui

class QCAWidget(QtGui.QWidget):
    ''' '''
    
    def __init__(self, parent=None):
        ''' '''
        super(QCAWidget, self).__init__(parent)
        
        self.initUI()
        
    def initUI(self):
        ''' '''
        pass