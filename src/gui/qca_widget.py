#!/usr/bin/python

# ---------------------------------------------------------
# Name: auxil.py
# Purpose: Auxiliary commonly used functions
# Author:    Jacob Retallick
# Created: 19.06.2014
# Last Modified: 19.06.2015
# ---------------------------------------------------------

from PyQt4 import QtGui, QtCore
import pylab as plt

import sys

# General parameters

# QCA cell parameters
CELL_SEP = 50
CELL_SIZE = 1.*CELL_SEP
CELL_ALPHA = int(.15*255)

# colors
DEFAULT_COLOR = QtGui.QColor(255, 255, 255)
INACTIVE_COLOR = QtGui.QColor(100, 100, 100)
OUTPUT_COLOR = QtGui.QColor(0, 200, 0, 150)
INPUT_COLOR = QtGui.QColor(200, 0, 0, 150)
FIXED_COLOR = QtGui.QColor(255, 165, 0, 150)

DOT_RAD = .25*CELL_SIZE
CELL_PEN_WIDTH = max(1, int(0.05*CELL_SIZE))
CELL_PEN_COLOR = QtGui.QColor(180, 180, 180)
TEXT_PEN_COLOR = QtGui.QColor(0, 0, 0)
TEXT_PEN_WIDTH = max(1, int(0.05*CELL_SIZE))
TEXT_FONT = QtGui.QFont('bold', 13)

MAX_MAG = 5.
MIN_MAG = 0.1
MAG_STEP = 0.1

SHOW_POLS = True

STYLES = {'blank': 0,
          'clocks': 1,
          'parts': 2,
          'pols': 3
          }


class QCACell:
    '''Class for QCA cell information'''

    def __init__(self, cell, spacing, offset, style='pols'):
        '''Construct a CellWidget from a cell dict, the grid spacing, and
        any desired offset (2 element iterable)'''

        self.x = CELL_SEP*(cell['x']-offset[0])*1./spacing
        self.y = CELL_SEP*(cell['y']-offset[1])*1./spacing

        self.cell_pen = QtGui.QPen(CELL_PEN_COLOR)
        self.cell_pen.setWidth(CELL_PEN_WIDTH)

        self.text_pen = QtGui.QPen(TEXT_PEN_COLOR)
        self.text_pen.setWidth(TEXT_PEN_WIDTH)

        self.type = cell['cf']
        self.qdots = []
        for qd in cell['qdots']:
            x = CELL_SEP*(qd['x']-offset[0])*1./spacing
            y = CELL_SEP*(qd['y']-offset[1])*1./spacing
            self.qdots.append([x, y])
        self.set_style(style)

        # optional parameters
        self.pol = None
        self.num = cell['num']
        self.clock = cell['clk']
        self.part = None
        self.num_parts = None

    def get_color(self):
        '''Determine the background color of the QCA Cell'''

        # first check for cell type
        if self.type == 0:
            if self.style == STYLES['clocks']:
                color = plt.cm.hsv(.2*(1+self.clock))
            elif self.style == STYLES['parts'] and self.part:
                if self.part < 0:
                    color = INACTIVE_COLOR
                else:
                    color = plt.cm.hsv((1+self.part)*1./self.num_parts)
            elif self.style == STYLES['pols'] and self.pol:
                color = plt.cm.bwr(.5*(1+float(self.pol)))
            else:
                color = DEFAULT_COLOR
        elif self.type == 1:    # input
            color = INPUT_COLOR
        elif self.type == 2:    # output
            color = OUTPUT_COLOR
        elif self.type == 3:    # fixed
            color = FIXED_COLOR
        else:
            print('Invalid cell type')
            color = DEFAULT_COLOR

        if type(color) is tuple:
            color = [int(255*c) for c in color]
            color[3] = CELL_ALPHA
            color = QtGui.QColor(*color)
        return color

    def set_pol(self, pol):
        '''Set the cell polarization'''
        self.pol = round(pol, 2)

    def set_part(self, part, num_parts):
        '''Set the partition index, keep track of number of partitions for
        cell coloring'''

        self.part = part
        self.num_parts = num_parts

    def set_style(self, style):
        ''' '''
        try:
            self.style = STYLES[style.lower()]
        except:
            self.style = STYLES['blank']

    def draw_cell(self, qp, scaling=1.):
        '''Draw the cell template at the objects x,y position'''

        qp.setPen(self.cell_pen)
        qp.setBrush(self.get_color())

        cell_rect = QtCore.QRect(scaling*(self.x-.5*CELL_SIZE),
                                 scaling*(self.y-.5*CELL_SIZE),
                                 scaling*CELL_SIZE,
                                 scaling*CELL_SIZE)
        qp.drawRect(cell_rect)

        for x, y in self.qdots:
            qp.drawEllipse(scaling*(x-.5*DOT_RAD),
                           scaling*(y-.5*DOT_RAD),
                           scaling*DOT_RAD,
                           scaling*DOT_RAD)

        if SHOW_POLS and not self.pol is None:
            qp.setPen(self.text_pen)
            qp.setFont(TEXT_FONT)
            qp.drawText(cell_rect, QtCore.Qt.AlignCenter, str((self.pol)))


class Canvas(QtGui.QWidget):
    '''Canvas to draw QCA cells on'''

    def __init__(self, parent):
        ''' '''
        super(Canvas, self).__init__()
        self.parent = parent
        self.scaling = 1.

    # Interrupts

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        for cell in self.parent.cells:
            cell.draw_cell(qp, self.scaling)
        qp.end()

    def rescale(self, zoom=True):
        '''Rescale convas'''
        geo = self.geometry()
        old_scaling = self.scaling
        if zoom:
            self.scaling = min(MAX_MAG, self.scaling + MAG_STEP)
        else:
            self.scaling = max(MIN_MAG, self.scaling - MAG_STEP)
        scale_fact = self.scaling/old_scaling
        geo.setWidth(geo.width()*scale_fact)
        geo.setHeight(geo.height()*scale_fact)
        self.setGeometry(geo)
        self.update()


class QCAWidget(QtGui.QScrollArea):
    '''Container class for drawing QCA Circuits'''

    def __init__(self, cells, spacing):
        '''Construct a QCAWidget object from a list of cells and the grid
        spacing (to normalise)'''

        super(QCAWidget, self).__init__()

        # mouse tracking
        self.mouse_pos = None

        # get minimum x and y position
        x_min = min([cell['x']-spacing for cell in cells])
        y_min = min([cell['y']-spacing for cell in cells])
        x_max = max([cell['x']-spacing for cell in cells])
        y_max = max([cell['y']-spacing for cell in cells])

        x_range = CELL_SEP*(2+(x_max-x_min)*1./spacing)
        y_range = CELL_SEP*(2+(y_max-y_min)*1./spacing)

        self.canvas = Canvas(self)
        self.canvas.setGeometry(0, 0, x_range, y_range)
        self.setWidget(self.canvas)

        offset = [x_min, y_min]

        self.cells = [QCACell(cell, spacing, offset)
                      for cell in cells]

    # Modifiers

    def set_pols(self, pols):
        '''Set cell polarizations'''

        for i in xrange(len(pols)):
            self.cells[i].set_pol(pols[i])

    def set_parts(self, parts):
        '''Set the partition indices for each cell'''

        # get number of non-negative partitions
        num_parts = len([i for i in set(parts) if i >= 0])

        for i in xrange(len(parts)):
            self.cells[i].set_part(parts[i], num_parts)

    def set_style(self, style):
        '''Set the coloring style for the cells'''

        for cell in self.cells:
            cell.set_style(style)

    # Interrupts

    def mousePressEvent(self, e):
        ''' '''
        self.mouse_pos = e.pos()

    def mouseMoveEvent(self, e):
        ''' '''
        if not self.mouse_pos is None:
            diff = e.pos() - self.mouse_pos
            self.mouse_pos = e.pos()
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - diff.y())
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - diff.x())

    def keyPressEvent(self, e):
        ''' '''
        if e.key() == QtCore.Qt.Key_Minus:
            self.canvas.rescale(zoom=False)
        elif e.key() == QtCore.Qt.Key_Plus:
            self.canvas.rescale(zoom=True)
        else:
            e.ignore()


def test_app(cells, spacing, pols=None, parts=None, style=None):
    '''Test application for showing just the QCA circuit'''

    app = QtGui.QApplication(sys.argv)

    w = QtGui.QMainWindow()
    w.setGeometry(100, 100, 600, 600)
    qca = QCAWidget(cells, spacing)
    w.setCentralWidget(qca)
    w.show()

    if not pols is None:
        qca.set_pols(pols)
    if not parts is None:
        qca.set_parts(parts)
    if not style is None:
        qca.set_style(style)

    app.exec_()
