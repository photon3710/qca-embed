#!/usr/bin/env python

#---------------------------------------------------------
# Name: coef_viewer.py
# Purpose: Simple viewer to h and J parameters
# Author:	Jacob Retallick
# Created: 12.08.2015
# Last Modified: 12.08.2015
#---------------------------------------------------------

from PyQt4 import QtGui, QtCore, QtSvg
from core.chimera import load_chimera_file
from core.chimera import linear_to_tuple

import sys
import os

WIN_X = 800
WIN_Y = 800

MAX_MAG = 5
MIN_MAG = 0.1
MAG_STEP = 0.1
MAG_WHEEL_FACT = 0.2

NODE_RAD = 0.07
NODE_OFFSET = 0.12
NODE_DELTA = 0.20

TILE_SIZE = 100
NODE_BORDER = 0.01*TILE_SIZE
BORDER_WIDTH = 0.02*TILE_SIZE
EDGE_WIDTH = 0.04*TILE_SIZE

chimera_file = '../bin/bay16.txt'

red = QtGui.QColor(255, 0, 0)
blue = QtGui.QColor(0, 0, 255)
black = QtGui.QColor(0, 0, 0)
background = QtGui.QColor(255, 255, 255)
blank = black

def shade(color, l):
    '''Change the color lightness'''
    h, s, _l, a = color.getHsl()
    new_color = QtGui.QColor()
    new_color.setHsl(h, s, l, a)
    return new_color

class Node:
    ''' '''
    
    def __init__(self, tile, h, l):
        ''' '''
        
        self.tile = tile
        self.h = h
        self.l = l
        self.tup = (tile.m, tile.n, h, l)
        
        self.hc = 0.
        self.Jc = {}
        
        if l < 2:
            self.z = NODE_OFFSET + self.l*NODE_DELTA
        else:
            self.z = 1 - NODE_OFFSET - (3-l)*NODE_DELTA
    
    def set_h(self, h):
        ''' '''
        self.hc = h
    
    def add_J(self, q, J):
        ''' '''
        self.Jc[q] = J

class TileWidget(QtGui.QWidget):
    ''' '''
    
    def __init__(self, m, n, parent=None):
        ''' '''
        
        super(TileWidget, self).__init__(parent)
        self.m = m
        self.n = n
        
        self.nodes = {}
    
    def set_nodes(self, adj):
        '''Reset the contained nodes using the full chimera adjacency'''
        
        # forget old nodes
        self.nodes = {}
        
        for h in [0, 1]:
            for l in range(4):
                key = (self.m, self.n, h, l)
                if key in adj and len(adj[key]) > 0:
                    self.nodes[(h, l)] = Node(self, h, l)
    
    def mousePressEvent(self, e):
        ''' '''
#        print('Clicked tile m:{0} n:{1}'.format(self.m, self.n))
        self.parent().mousePressEvent(e)
                    
class Canvas(QtGui.QWidget):
    ''' '''
    
    def __init__(self, parent=None):
        ''' '''
        super(Canvas, self).__init__(parent)
        self.initUI()
        
        # parameters
        self.scaling = 1.
        self.tiles = {}
        self.h = 1.
        self.w = 1.
        
    def initUI(self):
        ''' '''
        pass
    
    # PAINT FUNCTIONS
    
    def rescale(self, zoom=True, f=1.):
        ''' '''
        step = f*MAG_STEP
        if zoom:
            self.scaling = min(MAX_MAG, self.scaling + step)
        else:
            self.scaling = max(MIN_MAG, self.scaling - step)
        geo = self.geometry()
        geo.setWidth(self.w*self.scaling)
        geo.setHeight(self.h*self.scaling)
        self.setGeometry(geo)
        self.update()
    
    def move_nodes(self):
        ''' '''
        for m, n in self.tiles:
            tile = self.tiles[(m, n)]
            for h, l in tile.nodes:
                node = tile.nodes[(h, l)]
                if h:
                    x = .5*tile.width()
                    y = node.z*tile.height()
                else:
                    x = node.z*tile.width()
                    y = 0.5*tile.height()
                x -= NODE_RAD*tile.width() - tile.x()
                y -= NODE_RAD*tile.height() - tile.y()
                node.x = x
                node.y = y

    def draw_tiles(self, painter):
        for m, n in self.tiles:
            tile = self.tiles[(m, n)]
            pen = QtGui.QPen(black)
            pen.setWidth(BORDER_WIDTH*self.scaling)
            painter.setPen(pen)
            painter.setBrush(background)
            painter.drawRect(tile.geometry())
    
    def draw_edges(self, painter):
        
        for m1, n1 in self.tiles:
            tile = self.tiles[(m1, n1)]
            dxy = NODE_RAD*tile.width()
            for h1, l1 in tile.nodes:
                node1 = tile.nodes[(h1, l1)]
                for m2, n2, h2, l2 in node1.Jc:
                    node2 = self.tiles[(m2, n2)].nodes[(h2, l2)]
                    x1, y1 = node1.x+dxy, node1.y+dxy
                    x2, y2 = node2.x+dxy, node2.y+dxy
                    J = node1.Jc[(m2, n2, h2, l2)]
                    color = red if J > 0 else blue
                    light = 255*(1-abs(J)/2)
                    color = shade(color, light)
                    pen = QtGui.QPen(color)
                    pen.setWidth(EDGE_WIDTH)
                    painter.setPen(pen)
                    painter.drawLine(x1, y1, x2, y2)
    
    def draw_nodes(self, painter):
        for m, n in self.tiles:
            tile = self.tiles[(m, n)]
            for h, l in tile.nodes:
                node = tile.nodes[(h, l)]
                # get rect for drawing
                size = 2*NODE_RAD*tile.width()
                rect = QtCore.QRect(node.x, node.y, size, size)
                # set painting parameters
                light = 255*(1-abs(node.hc/2))
                pen = QtGui.QPen(black if abs(node.hc) > 0.05 else blank)
                brush = red if node.hc > 0 else blue if node.hc < 0 else background
                brush = shade(brush, light)
                painter.setPen(pen)
                painter.setBrush(brush)
                painter.drawEllipse(rect)
        
    def paint(self, painter):
        ''' '''
        self.move_nodes()
        self.draw_tiles(painter)
        self.draw_edges(painter)
        self.draw_nodes(painter)

    # EVENT HANDLERS
    def paintEvent(self, e):
        ''' '''
        painter = QtGui.QPainter()
        painter.begin(self)
        self.paint(painter)
        painter.end()

class ChimeraWidget(QtGui.QScrollArea):
    
    def __init__(self, parent=None):
        '''Construct a widget for viewing and assigned h and J parameters on
        the chimera architecture'''
        super(ChimeraWidget, self).__init__(parent)
        self.initUI()
        
        # parameters
        self.mouse_pos = None
        self.zoom_flag = False
        
        self.load_chimera(chimera_file)
        
        fname = QtGui.QFileDialog.getOpenFileName(self, os.getcwd())
        self.load_coefs(fname)
        
    def initUI(self):
        '''Initialise the user interface'''
        
        self.layout = QtGui.QGridLayout(self)
        self.canvas = Canvas(self)
        self.setWidget(self.canvas)
        self.canvas.setLayout(self.layout)
    
    def color_chimera(self, hc, Jc):
        '''Set up the nodes and edges of the chimera graph'''
        tiles = self.canvas.tiles
        # set h parameters
        for m, n, h, l in hc:
            try:
                tiles[(m, n)].nodes[(h, l)].set_h(hc[(m, n, h, l)])                
            except:
                print('No node {0} in chimera graph'.format(str((m, n, h, l))))
        # set J parameters
        for m, n, h, l in Jc:
            tiles[(m, n)].nodes[(h, l)].Jc = {}
            for qb in Jc[(m, n, h, l)]:
                tiles[(m, n)].nodes[(h, l)].add_J(qb, Jc[(m, n, h, l)][qb])

    def load_chimera(self, fname):
        '''Load and set the chimera structure'''
        
        try:
            M, N, adj = load_chimera_file(fname)
        except IOError:
            return
        
        self.M = M
        self.N = N

        self.adj = {linear_to_tuple(k, M, N):\
            [linear_to_tuple(a, M, N) for a in adj[k]] for k in adj}
        
        # forget old grid layout
        for tile in self.canvas.tiles:
            self.canvas.tiles[tile].setParent(None)
        self.canvas.tiles = {}

        while self.layout.count():
            item = self.layout.takeAt(0)
            item.widget().deleteLater()

        # resize canvas
        width = N*TILE_SIZE
        height = M*TILE_SIZE
        
        self.canvas.w = width
        self.canvas.h = height
        self.canvas.scaling = 1.
        self.canvas.setGeometry(0, 0, width, height)
        
        # set up tiles and default nodes
        for m in range(M):
            for n in range(N):
                tile = TileWidget(m, n, parent=self.canvas)
                tile.set_nodes(self.adj)
                self.layout.addWidget(tile, m, n)
                self.canvas.tiles[(m, n)] = tile
                tile.show()
        
        self.canvas.update()
    
    def load_coefs(self, fname):
        '''Load and set a coef file in the current chimera structure'''
        
        try:
            fp = open(fname)
        except:
            print('Failed to open coef file...')
            return
        
        # purge first line, don't need the number of qubits
        fp.readline()
        
        # extract h and J coefficients
        h = {}
        J = {}
        mapper = lambda qb: linear_to_tuple(qb, self.M, self.N, L=4)
        for line in fp:
            if len(line) < 3 or '#' in line:
                continue
            data = line.split()
            qb1 = int(data[0])
            qb2 = int(data[1])
            val = float(data[2])
            if qb1 == qb2:
                h[mapper(qb1)] = val
            else:
                if mapper(qb1) in J:
                    J[mapper(qb1)][mapper(qb2)] = val
                else:
                    J[mapper(qb1)] = {mapper(qb2): val}
        fp.close()
        
        # color chimera graph
        self.color_chimera(h, J)
        
        self.canvas.update()
    
    def save_canvas(self):
        ''' '''
        
        fname = QtGui.QFileDialog.getSaveFileName(self, 'Save SVG file...')
        
        if len(fname) == 0:
            return

        generator = QtSvg.QSvgGenerator()
        generator.setFileName(fname)
        generator.setSize(self.canvas.size())
        generator.setViewBox(self.canvas.rect())
        
        painter = QtGui.QPainter()
        painter.begin(generator)
        self.canvas.paint(painter)
        painter.end()
    
    # EVENT HANDLERS
    
    def mousePressEvent(self, e):
        '''Store mouse position on press'''
        self.mouse_pos = e.pos()
    
    def mouseMoveEvent(self, e):
        '''If pressed, drag scroll area'''
        
        if self.mouse_pos is not None:
            diff = e.pos()-self.mouse_pos
            self.mouse_pos = e.pos()
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value()-diff.y())
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value()-diff.x())
    
    def mouseReleaseEvent(self, e):
        '''Release mouse position on release'''
        self.mouse_pos = None
    
    def keyPressEvent(self, e):
        ''' '''
        if e.key() == QtCore.Qt.Key_Minus:
            self.canvas.rescale(zoom=False)
        elif e.key() == QtCore.Qt.Key_Plus:
            self.canvas.rescale(zoom=True)
        elif e.key() == QtCore.Qt.Key_Control:
            self.zoom_flag = True
        elif e.key() == QtCore.Qt.Key_S:
            self.save_canvas()
        else:
            e.ignore()

    def keyReleaseEvent(self, e):
        ''' '''
        if e.key() == QtCore.Qt.Key_Control:
            self.zoom_flag = False
    
    def wheelEvent(self, e):
        '''If holding'''
        
        if self.zoom_flag:
            self.canvas.rescale(zoom=e.delta() > 0, f=MAG_WHEEL_FACT)
        else:
            super(ChimeraWidget, self).wheelEvent(e)


def main():
    '''Main loop which initialises embedder application'''

    app = QtGui.QApplication(sys.argv)

    w = QtGui.QMainWindow()
    w.setGeometry(100, 100, WIN_X, WIN_Y)
    
    chimera_widget = ChimeraWidget(w)
    w.setCentralWidget(chimera_widget)

    w.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

    