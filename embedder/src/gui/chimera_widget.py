#!/usr/bin/env python

# -----------------------------------
# Name: chimera_widget.py
# Desc: ChimeraWidget class definition
# Author: Jake Retallick
# Created: 2015.11.25
# Modified: 2015.11.25
# Licence: Copyright 2015
# -----------------------------------

from PyQt4 import QtGui, QtCore
import gui_settings as settings


class ChimeraNode(QtGui.QWidget):
    '''Graph node representing a qubit on D-Wave's processor'''

    def __init__(self, tile, h, l, active):
        '''inputs:  tile    - ChimeraTile object containing the node
                    h       - flag for horizontal qubit
                    l       - index of qubit in tile (0...3)
        '''

        super(ChimeraNode, self).__init__(tile)

        self.tile = tile
        self.active = active    # flag determines whether node is active
        self.h = h
        self.l = l
        self.n = 4*(0 if h else 1)+l    # index of node within tile

        # determine center position within tile
        if l < 2:
            self.z = settings.CHIMERA_NODE_OFFSET + \
                settings.CHIMERA_NODE_DELTA*l
        else:
            self.z = 1-settings.CHIMERA_NODE_OFFSET - \
                settings.CHIMERA_NODE_DELTA*(3-l)

    def getColor(self):
        '''Return node color'''

        if not self.active:
            color = settings.CHIMERA_COL['inactive']
        else:
            color = settings.CHIMERA_COL['active']
        return color

    def drawNode(self, painter):
        '''Draw the node within the tile'''

        # determine center point of node
        if self.h:
            x = .5*self.tile.width()
            y = self.z*self.tile.height()
        else:
            y = .5*self.tile.height()
            x = self.z*self.tile.width()

        # shift to top left corner
        x = x-settings.CHIMERA_NODE_RAD*self.tile.width()
        y = y-settings.CHIMERA_NODE_RAD*self.tile.width()

        size = 2*settings.CHIMERA_NODE_RAD*self.tile.width()

        self.setGeometry(x, y, size, size)
        rect = QtCore.QRect(x+self.tile.x(), y+self.tile.y(),
                            size, size)
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 255))
        pen.setWidth(settings.CHIMERA_PEN_WIDTH)
        brush = QtGui.QBrush(self.getColor())

        painter.setPen(pen)
        painter.setBrush(brush)

        painter.drawEllipse(rect)

    def mousePressEvent(self, e):
        print('Node clicked...')
        self.tile.parent.onNodeClick(
            self.tile.m, self.tile.n, self.h, self.l)


class ChimeraTile(QtGui.QWidget):
    ''' '''

    def __init__(self, parent, m, n, adj=None):
        '''Tile in the chimera graph'''

        super(ChimeraTile, self).__init__(parent)

        self.parent = parent    # ChimeraWidget
        self.m = m              # tile row
        self.n = n              # tile column
        self.adj = adj          # internal adjacency list

        self.mouse_pos = None
        self.nodes = {}
        self.selected = False

        # initialise nodes
        for h in [True, False]:
            for l in xrange(4):
                n = 4*(1 if h else 0)+l
                active = len(adj[n]) > 0
                if active:
                    node = ChimeraNode(self, h, l, active)
                    self.nodes[(h, l)] = node
                    node.show()

    def mousePressEvent(self, e):
        '''On mouse click, store mouse position to check release'''
        self.mouse_pos = e.pos()
        self.parent.mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        '''On mouse release, if same tile, select tile for embedding'''
        if e.button() == QtCore.Qt.RightButton:
            self.parent.releaseSelection()
        elif self.mouse_pos is not None:
            diff = e.pos()-self.mouse_pos
            if max(abs(diff.x()), abs(diff.y())) < self.width():
                # same tile release, select tile pass to
                self.parent.onTileClick(self.m, self.n)

    def getColor(self):
        '''Get the color of the tile'''
        if self.selected:
            return settings.CHIMERA_COL['tile-selected']
        else:
            return settings.CHIMERA_COL['tile']

    def drawTile(self, painter):
        '''Draw tile background'''

        # outline
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 255))
        brush = QtGui.QBrush(self.getColor())

        painter.setPen(pen)
        painter.setBrush(brush)

        painter.drawRect(self.geometry())

        painter.setFont(QtGui.QFont('Decorative',
                                    settings.CHIMERA_FONT_SIZE))
        painter.drawText(
            self.x()+settings.CHIMERA_LABEL_OFFSET,
            self.y()+2*settings.CHIMERA_LABEL_OFFSET,
            '{0}:{1}'.format(self.m, self.n))

    def drawNodes(self, painter):
        '''Draw nodes within the tile'''

        for key in self.nodes:
            node = self.nodes[key]
            node.drawNode(painter)

    def drawEdges(self, painter):
        '''Draw connectiones between nodes'''

        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 255))
        pen.setWidth(settings.CHIMERA_EDGE_WIDTH)
        painter.setPen(pen)

        dxy = settings.CHIMERA_NODE_RAD*self.width()
        for i in xrange(7):
            n1 = self.nodes[divmod(i, 4)]
            x1 = self.x()+n1.x()+dxy
            y1 = self.y()+n1.y()+dxy
            for j in self.adj[i]:
                n2 = self.nodes[divmod(j, 4)]
                x2 = self.x()+n2.x()+dxy
                y2 = self.y()+n2.y()+dxy
                painter.drawLine(x1, y1, x2, y2)


class Canvas(QtGui.QWidget):
    ''' '''

    def __init__(self, parent):
        '''Initialise Canvas'''
        super(Canvas, self).__init__(parent)
        self.parent = parent
        self.scaling = 1.

    def drawTiles(self, painter):
        '''Draw nodes and connectors within tiles'''

        for key in self.parent.tiles:
            m, n = key
            tile = self.parent.tiles[key]
            tile.drawTile(painter)

    def drawExternConnectors(self, painter):
        '''Draw connectors between tiles'''
        pass
        # still needs to be implemented

    def drawInterns(self, painter):
        ''' '''

        for key in self.parent.tiles:
            m, n = key
            tile = self.parent.tiles[key]
            tile.drawEdges(painter)
            tile.drawNodes(painter)

    def paintEvent(self, e):
        painter = QtGui.QPainter()
        painter.begin(self)

        # draw tile background
        self.drawTiles(painter)

        # draw external connector
        self.drawExternConnectors(painter)

        # draw tile connects
        self.drawInterns(painter)

        painter.end()


class ChimeraWidget(QtGui.QScrollArea):
    '''Widget for viewing QCA circuits'''

    def __init__(self):
        ''' '''
        super(ChimeraWidget, self).__init__()

        # parameters
        self.shift = False          # flag for shift pressed
        self.clicked_tile = None    # pair of indices for selected tile
        self.active_graph = None    # graph into which to run embedding
        self.tiles = {}

        self.mouse_pos = None

        self.initUI()

        self.updateChimera(2)   # test: 2 should force error when implemented

    def initUI(self):
        '''Initialise UI'''

        self.layout = QtGui.QGridLayout(self)
        self.canvas = Canvas(self)
        self.setWidget(self.canvas)
        self.canvas.setLayout(self.layout)

    def updateChimera(self, filename):
        '''Process a chimera specification file and update the widget'''

        # placeholder code until chimera read functionality implemented
        M = 12
        N = 12

        # forget old grid layout
        for tile in self.tiles:
            tile.setParent(None)
        self.tiles = {}

        while self.layout.count():
            item = layout.takeAt(0)
            item.widget().deleteLater()

        # resize canvas
        width = N*settings.CHIMERA_TILE_SIZE
        height = M*settings.CHIMERA_TILE_SIZE

        self.canvas.setGeometry(0, 0, width, height)

        adj = [[4, 5, 6, 7]]*4 + [[0, 1, 2, 3]]*4
        for m in xrange(M):
            for n in xrange(N):
                tile = ChimeraTile(self, m, n, adj=adj)
                self.tiles[(m, n)] = tile
                self.layout.addWidget(tile, m, n)
                tile.show()

        self.canvas.update()

    def onTileClick(self, m, n):
        '''If a tile is clicked and shift-flag False, start subgraph
        select. If a tile is clicked and shift-flag True, end subgraph
        select.'''

        if self.clicked_tile is not None and self.shift:
            # select subgraph
            if self.clicked_tile != (m, n):
                self.active_graph = self.getActiveGraph(
                    self.clicked_tile, (m, n))
                self.canvas.update()
        else:
            # first corner of subgraph
            self.clicked_tile = (m, n)
            # unselected all other tiles
            for key in self.tiles:
                self.tiles[key].selected = False
            # update selected subgraph
            self.active_graph = self.getActiveGraph((m, n), None)
            self.canvas.update()

    def onNodeClick(self, m, n, h, l):
        '''On node click'''
        pass

    def getActiveGraph(self, tile1, tile2):
        '''Isolate the subgraph between the selected corners'''

        # find range of selected tiles
        if tile2 is None:
            ry = [tile1[0], tile1[0]]
            rx = [tile1[1], tile1[1]]
        else:
            ry = [min(tile1[0], tile2[0]), max(tile1[0], tile2[0])]
            rx = [min(tile1[1], tile2[1]), max(tile1[1], tile2[1])]

        # for now just select tiles
        for m in xrange(ry[0], ry[1]+1):
            for n in xrange(rx[0], rx[1]+1):
                print m, n
                self.tiles[(m, n)].selected = True

    def releaseSelection(self):
        ''' '''

        for key in self.tiles:
            self.tiles[key].selected = False

        self.clicked_tile = None
        self.active_graph = None

        self.canvas.update()

    def keyPressEvent(self, e):
        ''' '''

        if e.key() == QtCore.Qt.Key_Shift:
            self.shift = True

    def keyReleaseEvent(self, e):
        '''Reset key flags'''

        self.shift = False

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

    def mouseReleaseEvent(self, e):
        '''On mouse release, forget old mouse position to avoid
        jumping. On right click release unselect everything'''
        self.mouse_pos = None
