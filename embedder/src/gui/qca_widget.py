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
from core.parse_qca import parse_qca_file

import numpy as np


class QCACellWidget(QtGui.QWidget):
    '''QCA Cell Widget'''

    def __init__(self, parent, cell, spacing=1., offset=[0, 0]):
        ''' '''
        super(QCACellWidget, self).__init__(parent)

        self.parent = parent
        self.qca_widget = parent.parent

        # cell parameters
        self.x = settings.CELL_SEP*(cell['x']-offset[0])*1./spacing
        self.y = settings.CELL_SEP*(cell['y']-offset[1])*1./spacing

        self.cell_pen = QtGui.QPen(settings.CELL_PEN_COLOR)
        self.cell_pen.setWidth(settings.CELL_PEN_WIDTH)

        self.text_pen = QtGui.QPen(settings.TEXT_PEN_COLOR)
        self.text_pen.setWidth(settings.TEXT_PEN_WIDTH)

        self.type = cell['cf']
        self.qdots = []
        self.num = cell['number']
        for qd in cell['qdots']:
            x = settings.CELL_SEP*(qd['x']-offset[0])*1./spacing
            y = settings.CELL_SEP*(qd['y']-offset[1])*1./spacing
            self.qdots.append([x, y])

        self.setGeometry(0, 0, settings.CELL_SIZE, settings.CELL_SIZE)
        self.clicked = False

    def get_color(self):
        '''Determine the background color of the QCA Cell'''

        # check if selected
        if self.clicked:
            return settings.QCA_COL['clicked']
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

    def drawCell(self, painter):
        '''Redraw cell widget'''

        painter.setPen(self.cell_pen)
        painter.setBrush(self.get_color())

        painter.drawRect(self.geometry())

        # write cell label
        painter.setPen(self.text_pen)
        painter.setFont(QtGui.QFont('Decorative', 10))
        painter.drawText(self.geometry(), QtCore.Qt.AlignCenter, str(self.num))

    def mousePressEvent(self, e):
        ''' '''
#        print('Clicked cell {0}'.format(self.num))
        self.qca_widget.onClick(self.num)


class Canvas(QtGui.QWidget):
    '''Canvas to draw QCA cells'''

    def __init__(self, parent):
        ''' '''
        super(Canvas, self).__init__()
        self.parent = parent
        self.scaling = 1.

        self.w = 1.
        self.h = 1.

    # Interrupts
    def paintEvent(self, e):
        ''' '''
        painter = QtGui.QPainter()
        painter.begin(self)
        self.moveCells()
        self.drawConnections(painter)
        self.drawCells(painter)
        painter.end()

    def mousePressEvent(self, e):
        self.parent.mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        ''' '''
        pass
        # determine which cell was clicked

    def rescale(self, zoom=True, f=1.):
        ''' '''
        step = f*settings.MAG_STEP
        if zoom:
            self.scaling = min(settings.MAX_MAG, self.scaling + step)
        else:
            self.scaling = max(settings.MIN_MAG, self.scaling - step)
        geo = self.geometry()
        geo.setWidth(self.w*self.scaling)
        geo.setHeight(self.h*self.scaling)
        self.setGeometry(geo)
        self.update()

    def moveCells(self):
        ''' '''
        for cell in self.parent.cells:
            _x = cell.x*self.scaling
            _y = cell.y*self.scaling
            _size = settings.CELL_SIZE*self.scaling
            cell.setGeometry(_x, _y, _size, _size)

    def drawCells(self, painter):
        ''' '''
        for cell in self.parent.cells:
            cell.drawCell(painter)

    def drawConnections(self, painter):
        ''' '''
        J0 = self.parent.J0

        # store cell centers
        X = []
        Y = []
        for cell in self.parent.cells:
            geo = cell.geometry()
            X.append(geo.x()+.5*geo.width())
            Y.append(geo.y()+.5*geo.height())
        # draw all non-zero interactions
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 255))
        pen.setWidth(max(1, settings.INT_PEN_WIDTH*self.scaling))
        for i in xrange(J0.shape[0]-1):
            for j in xrange(i+1, J0.shape[0]):
                if abs(J0[i, j]) > 0.:
                    pen.setStyle(settings.INT_PEN_STYLE[
                        'strong' if abs(J0[i, j]) > 0.5 else 'weak'])
                    painter.setPen(pen)
                    painter.drawLine(X[i], Y[i], X[j], Y[j])


class QCAWidget(QtGui.QScrollArea):
    '''Widget for viewing QCA circuits'''

    def __init__(self, filename=None):
        ''' '''
        super(QCAWidget, self).__init__()

        # parameters
        self.cells = []         # list of qca cells
        self.spacing = 1.       # cell-cell spacing value
        self.J = np.zeros(0)    # cell-cell interaction matrix
        self.J0 = self.J        # J with only included interactions

        # mouse tracking
        self.mouse_pos = None
        self.initUI()

        if filename is not None:
            self.updateCircuit(filename)

    def initUI(self):
        ''' '''

        # parameters
        self.mouse_pos = None
        self.cells = []
        self.zoom_flag = False

        # create main widget
        self.canvas = Canvas(self)
        self.canvas.setGeometry(0, 0, 0, 0)
        self.setWidget(self.canvas)

    def updateCircuit(self, filename):
        ''' '''

        try:
            cells, spacing, zones, J, feedback = \
                    parse_qca_file(filename, one_zone=True)
        except:
            print('Failed to load QCA File...')
            return

        # forget old circuit
        self.cells = []

        # update J coefficients
        self.J = J
        self.J0 = J/np.max(np.abs(J))

        # find span and offset of circuit: currently inefficient
        x_min = min([cell['x'] for cell in cells])
        x_max = max([cell['x'] for cell in cells])
        y_min = min([cell['y'] for cell in cells])
        y_max = max([cell['y'] for cell in cells])

        o = settings.QCA_CANVAS_OFFSET
        span = [x_max-x_min, y_max-y_min]
        offset = [x_min-o*span[0], y_min-o*span[1]]

        # update circuit constants
        self.spacing = spacing
        self.offset = offset

        # update size and scaling of canvas
        factor = (1+2*o)*settings.CELL_SEP*1./spacing
        self.canvas.scaling = 1.
        self.canvas.w = span[0]*factor
        self.canvas.h = span[1]*factor
        self.canvas.setGeometry(0, 0,
                                self.canvas.w, self.canvas.h)

        # add new cells
        for cell in cells:
            self.addCell(cell)

        self.canvas.update()

    def addCell(self, cell):
        ''' '''
        cell = QCACellWidget(self.canvas, cell,
                             spacing=self.spacing, offset=self.offset)
        self.cells.append(cell)
        cell.show()

    def onClick(self, num):
        '''Response to clicking on one the QCA cells'''

        for cell in self.cells:
            cell.clicked = False
        self.cells[num].clicked = True

        self.canvas.update()
        print('Detected click on cell {0}'.format(num))

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
