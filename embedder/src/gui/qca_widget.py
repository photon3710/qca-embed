#!/usr/bin/env python

# -----------------------------------
# Name: qca_widget.py
# Desc: QCAWidget class definition
# Author: Jake Retallick
# Created: 2015.11.25
# Modified: 2015.11.25
# Licence: Copyright 2015
# -----------------------------------

from PyQt4 import QtGui, QtCore
import gui_settings as settings


class QCACellWidget(QtGui.QWidget):
    '''QCA Cell Widget'''

    def __init__(self, parent, x, y, dx, dy):
        ''' '''
        super(QCACellWidget, self).__init__(parent)
        self.setGeometry(x, y, dx, dy)
        

    def paintEvent(self, e):
        ''' '''
        painter = QtGui.QPainter()
        painter.begin(self)
        self.drawCell(painter)
        painter.end()

    def drawCell(self, painter, scaling=1.):
        ''' '''
        painter.setPen(QtGui.QColor(0, 0, 0))
        rect = self.geometry()
        painter.drawRect(rect)


class Canvas(QtGui.QWidget):
    '''Canvas to draw QCA cells'''

    def __init__(self, parent):
        ''' '''
        super(Canvas, self).__init__()
        self.parent = parent
        self.scaling = 1.

    # Interrupts
    def paintEvent(self, e):
        ''' '''
        painter = QtGui.QPainter()
        painter.begin(self)
        self.drawCircuit(painter)
        painter.end()

    def rescale(self, zoom=True):
        ''' '''
        geo = self.geometry()
        old_scaling = self.scaling
        if zoom:
            self.scaling = min(settings.MAX_MAG,
                               self.scaling + settings.MAG_STEP)
        else:
            self.scaling = max(settings.MIN_MAG,
                               self.scaling - settings.MAG_STEP)
        scale_fact = self.scaling/old_scaling
        geo.setWidth(geo.width()*scale_fact)
        geo.setHeight(geo.height()*scale_fact)
        self.setGeometry(geo)
        self.update()

    def drawCircuit(self, painter):
        ''' '''
        for cell in self.parent.cells:
            cell.drawCell(painter, self.scaling)


class QCAWidget(QtGui.QScrollArea):
    '''Widget for viewing QCA circuits'''

    def __init__(self, filename=None):
        ''' '''
        super(QCAWidget, self).__init__()

        # mouse tracking
        self.mouse_pos = None
        self.cells = []     # list of qca cells
        self.initUI()

    def initUI(self):
        ''' '''

        # parameters
        self.mouse_pos = None
        self.cells = []

        # create main widget
        self.canvas = Canvas(self)
        self.canvas.setGeometry(0, 0, 1000, 1000)
        self.setWidget(self.canvas)

        self.addCell(60, 10, settings.CELL_SIZE, settings.CELL_SIZE)
        self.addCell(100, 100, settings.CELL_SIZE, settings.CELL_SIZE)

#    def paintEvent(self, e):
#        ''' '''
#        super(QCAWidget, self).paintEvent(e)
#        painter = QtGui.QPainter()
#        painter.begin(self)
#        self.drawCircuit(painter)
#        painter.end()

    def addCell(self, x, y, dx, dy):
        ''' '''
        cell = QCACellWidget(self, x, y, dx, dy)
        self.cells.append(cell)

    # interrupts

    def mousePressEvent(self, e):
        '''On left click drag circuit, on right click highlight cell'''
        self.mouse_pos = e.pos()

    def mouseMoveEvent(self, e):
        ''' '''
        if self.mouse_pos is not None:
            diff = e.pos()-self.mouse_pos
            self.mouse_pos = e.pos()
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value()-diff.y())
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value()-diff.x())

    def keyPressEvent(self, e):
        ''' '''
        if e.key() == QtCore.Qt.Key_Minus:
            self.canvas.rescale(zoom=False)
        elif e.key() == QtCore.Qt.Key_Plus:
            self.canvas.rescale(zoom=True)
        else:
            e.ignore()
