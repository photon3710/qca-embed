#!/usr/bin/env python

# -----------------------------------
# Name: chimera_widget.py
# Desc: ChimeraWidget class definition
# Author: Jake Retallick
# Created: 2015.11.25
# Modified: 2015.11.25
# Licence: Copyright 2015
# -----------------------------------

from PyQt4 import QtGui


class ChimeraWidget(QtGui.QScrollArea):
    '''Widget for vieweing QCA circuits'''

    def __init__(self):
        ''' '''
        super(ChimeraWidget, self).__init__()