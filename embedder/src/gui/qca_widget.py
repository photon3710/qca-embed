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

    def __init__(self, parent, cell, spacing=1., offset=[0, 0]):
        ''' '''
        super(QCACellWidget, self).__init__(parent)

        # cell parameters
        self.x = settings.CELL_SEP*(cell['x']-offset[0])*1./spacing
        self.y = settings.CELL_SEP*(cell['y']-offset[1])*1./spacing

        self.cell_pen = QtGui.QPen(settings.CELL_PEN_COLOR)
        self.cell_pen.setWidth(settings.CELL_PEN_WIDTH)

        self.text_pen = QtGui.QPen(settings.TEXT_PEN_COLOR)
        self.text_pen.setWidth(settings.TEXT_PEN_WIDTH)

        self.type = cell['cf']
        self.qdots = []
        for qd in cell['qdots']:
            x = settings.CELL_SEP*(qd['x']-offset[0])*1./spacing
            y = settings.CELL_SEP*(qd['y']-offset[1])*1./spacing
            self.qdots.append([x, y])

    def get_color(self):
        '''Determine the background color of the QCA Cell'''

        # first check for cell type
        if self.type == 0:
            color = settings.QCA_COL['default']
        elif self.type == 1:    # input
            color = settings.QCA_COL['input']
        elif self.type == 2:    # output
            color = settings.QCA_COL['output']
        elif self.type == 3:    # fixed
            color = settings.QCA_COL['fixed']
        else:
            print('Invalid cell type')
            color = settings.QCA_COL['default']

        if type(color) is tuple:
            color = [int(255*c) for c in color]
            color[3] = settings.CELL_ALPHA
            color = QtGui.QColor(*color)
        return color

    def paintEvent(self, e):
        ''' '''
        painter = QtGui.QPainter()
        painter.begin(self)
        self.drawCell(painter)
        painter.end()

    def drawCell(self, painter, scaling=1.):
        '''Move and repaint cell widget'''
        painter.setPen(self.pen)
        painter.setBrush(self.get_color())
        rect = self.geometry()
        painter.drawRect(rect)

        # draw dots
        pass


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

    def rescale(self, zoom=True, f=1.):
        ''' '''
        geo = self.geometry()
        old_scaling = self.scaling
        step = f*settings.MAG_STEP
        if zoom:
            self.scaling = min(settings.MAX_MAG, self.scaling + step)
        else:
            self.scaling = max(settings.MIN_MAG, self.scaling - step)
        scale_fact = self.scaling/old_scaling
        geo.setWidth(geo.width()*scale_fact)
        geo.setHeight(geo.height()*scale_fact)
        self.setGeometry(geo)
        self.update()

    def drawCircuit(self, painter):
        ''' '''
        cells = self.parent.cells
        for cell in cells:
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
        self.zoom_flag = False

        # create main widget
        self.canvas = Canvas(self)
        self.canvas.setGeometry(0, 0, 1000, 1000)
        self.setWidget(self.canvas)

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
        elif e.key() == QtCore.Qt.Key_Control:
            self.zoom_flag = True
        else:
            e.ignore()

    def keyReleaseEvent(self, e):
        ''' '''
        if e.key() == QtCore.Qt.Key_Control:
            self.zoom_flag = False

    def wheelEvent(self, e):
        '''Scrolling options'''

        if self.zoom_flag:
            self.canvas.rescale(zoom=e.delta() > 0, f=settings.MAG_WHEEL_FACT)
        else:
            super(QCAWidget, self).wheelEvent(e)
