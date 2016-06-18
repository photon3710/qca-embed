#!/usr/bin/env python

# -----------------------------------
# Name: chimera_widget.py
# Desc: Viewer widget for Chimera graph
# Author: Jake Retallick
# Created: 2015.12.15
# Modified: 2015.12.15
# Licence: Copyright 2015
# -----------------------------------

from PyQt4 import QtGui

class ChimeraWidget(QtGui.QWidget):
    ''' '''
    
    def __init__(self, parent=None):
        ''' '''
        super(ChimeraWidget, self).__init__(parent)
        
        self.initUI()
        
    def initUI(self):
        ''' '''
        pass