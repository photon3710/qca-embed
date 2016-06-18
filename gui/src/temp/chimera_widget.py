from PyQt4 import QtGui, QtCore

from core.chimera import load_chimera_file, linear_to_tuple

# magnification parameters
MAX_MAG = 5
MIN_MAG = 0.1
MAG_STEP = 0.1
MAG_WHEEL_FACT = 0.2

# node drawing parameters
NODE_RAD = 0.07
NODE_OFFSET = 0.12
NODE_DELTA = 0.20

# drawing parameters
TILE_SIZE = 100
NODE_BORDER = 0.01*TILE_SIZE
BORDER_WIDTH = 0.02*TILE_SIZE
EDGE_WIDTH = 0.04*TILE_SIZE



def lum_color(color, l):
    '''Change the color luminance: l in [0,1] with 0->black, 1->white'''
    if not isinstance(color, QtGui.QColor):
        return color
    h, s, _, a = color.getHslF()
    new_color = QtGui.QColor()
    new_color.setHslF(h, s, l, a)
    return new_color

# colors
BLACK = QtGui.QColor(0,0,0)
WHITE = QtGui.QColor(255, 255, 255)
NODE_BKG = lum_color(WHITE, .7)     # default background color for nodes
TILE_BKG = lum_color(WHITE, .5)     # default background color for tiles


class ChimeraNode(QtGui.QWidget):
    ''' '''
    
    def __init__(self, tile, h, l):
        '''Initialise a Node object'''
        super(ChimeraNode, self).__init__(tile)
        
        assert isinstance(tile, ChimeraTile), 'Must specifiy ChimeraTile parent'
        
        # pointers
        self.tile = tile
        self.chimera_widget = tile.chimera_widget

        # parameters
        self.h = h      # horizontal qubit flag
        self.l = l      # index in sub-tile
        self.tup = (tile.m, tile.n, h, l)   # full address for qubit
        
        if l < 2:
            self.z = NODE_OFFSET + self.l*NODE_DELTA
        else:
            self.z = 1 - NODE_OFFSET - (3-l)*NODE_DELTA
        
        self.mouse_pos = None
        self.node_color_map = None   # mapping to determine node color
        self.node_color_par = None   # extra arguments for color_map
    
    # paint methods
    
    def get_color(self):
        '''Determine background color for node'''
        
        if self.color_map is None:
            color = NODE_BKG
        else:
            color = self.node_color_map(self, **self.node_color_par)
        return color

    def draw_node(self, painter):
        '''Paint the node'''
        
        painter.setPen(BLACK)
        painter.setBrush(self.get_color())
        painter.drawEllipse(self.geometry())
        
    def paintEvent(self, e):
        ''' '''
        painter = QtGui.QPainter()
        painter.begin(self)
        self.draw_node(painter)
        painter.end()
    
    # Interrupts
    
    def mousePressEvent(self, e):
        '''On click, store current mouse position'''
        self.mouse_pos = e.pos()
    
    def mouseReleaseEvent(self, e):
        '''On release, trigger node press if cursor hasn't significantly
        moved'''
        if self.mouse_pos:
            diff = e.pos()-self.mouse_pos
            self.mouse_pos = None
            if max(abs(diff.x()), abs(diff.y())) < self.width():
                self.chimera_widget.on_node_click(self.tup, e)
        

class ChimeraTile(QtGui.QWidget):
    '''Simple class containing some number of ChimeraNodes'''
    
    def __init__(self, chimera, m, n):
        '''Initialise a Chimera Tile object'''
        super(ChimeraTile, self).__init__(chimera)
        
        assert isinstance(chimera, ChimeraWidget), 'Must specifiy ChimeraWidget'
        self.chimera_widget = chimera

        self.m = m  # row index of tile
        self.n = n  # column index of tile
        self.L = 4  # characteristic number of qubits: horizontal or vertical
        self.nodes = {}     # container for all ChimeraNode objects

        self.mouse_pos = None
        self.tile_color_map = None   # mapping to determine tile color
        self.tile_color_par = None   # external parameters for color_map

    def set_nodes(self, adj):
        '''Reset the contained nodes using the complete chimera adjacency dict'''
        
        # forget old nodes
        for node in self.nodes:
            node.setParent(None)
        self.nodes = {}
        
        for h in [0, 1]:
            for l in range(self.L):
                key = (self.m, self.n, h, l)
                if key in adj and len(adj[key]) > 0:
                    self.nodes[(h, l)] = ChimeraNode(self, h, l)
    
    # painting
    
    def get_color(self):
        '''Determine the tile background color'''
        
        if self.color_map is None:
            color = TILE_BKG
        else:
            color = self.tile_color_map(self, **self.tile_color_par)
        return color
    
    def mousePressEvent(self, e):
        '''On tile click, store current cursor position'''
        self.mouse_pos = e.pos()
        
    def mouseReleaseEvent(self, e):
        '''On release, trigger tile press if cursor hasn't significantly moved'''
        if self.mouse_pos:
            diff = e.pos()-self.mouse_pos
            self.mouse_pos = None
            if max(abs(diff.x()), abs(diff.y())) < self.width():
                self.chimera_widget.on_node_click(self.tup, e)


class Canvas(QtGui.QWidget):
    '''Drawing canvas for Chimera Widget'''
    
    def __init__(self, chimera):
        '''Initialise Canvas object'''
        super(Canvas, self).__init__(chimera)
        
        assert isinstance(chimera, ChimeraWidget), 'Must specify ChimeraWidget'
        self.initUI()

        # parameters
        self.scaling = 1.   # scaling factor
        self.h = 1.         # width for scaling=1
        self.w = 1.         # height for scaling=1
        
        self.tiles = {}     # container for ChimeraTiles
        self.edge_color_map = None   # mapping for edge colors
        self.edge_color_pars = None  # dict of external parameters for edge colors
        
    def initUI(self):
        pass
    
    # PAINT FUNCTIONS
    
    def rescale(self, zoom=True, f=1.):
        '''Rescale the canvas'''
        
        step = f*MAG_STEP
        if zoom:
            self.scaling = min(MAX_MAG, self.scaling+step)
        else:
            self.scaling = max(MIN_MAG, self.scaling-step)
        geo = self.geometry()
        geo.setWidth(self.w*self.scaling)
        geo.setHeight(self.h*self.scaling)
        self.setGeometry(geo)
        self.update()
        
    def move_nodes(self):
        '''Update the ChimeraNode geometries'''
        
        for _, tile in self.tiles.items():
            for h, l in tile.nodes:
                node = tile.nodes[(h, l)]
                if h:
                    x = .5*tile.width()
                    y = node.z*tile.height()
                else:
                    x = node.z*tile.width()
                    y = 0.5*tile.height()
                x += tile.x() - NODE_RAD*tile.width()
                y += tile.y() - NODE_RAD*tile.height()
                size = 2*NODE_RAD*tile.width()
                node.setGeometry(x, y, size, size)
    
    def draw_tiles(self, painter):
        '''Draw tile background'''
        
        for kt, tile in self.tiles.items():
            pen = QtGui.QPen(BLACK)
            pen.setWidth(BORDER_WIDTH*self.scaling)
            painter.setPen(pen)
            painter.setBrush(tile.get_color())
            painter.drawRect(tile.geometry())
    
    def draw_edges(self, painter):
        '''Draw all edges'''
        pass
    
    def draw_nodes(self, painter):
        '''Draw all nodes'''
        pass
    
    def draw_labels(self, painter):
        '''Draw tile labels'''
        pass
    
    def paint(self, painter):
        '''Draw everything'''
        self.move_nodes()
        self.draw_tiles(painter)
        self.draw_edges(painter)
        self.draw_nodes(painter)
        self.draw_labels(painter)

    def paintEvent(self, e):
        painter = QtGui.QPainter()
        painter.begin(self)
        self.paint(painter)
        painter.end()


class ChimeraWidget(QtGui.QScrollArea):
    '''Widget for displaying structure in the Chimera graph'''
    
    def __init_(self, parent=None):
        '''Initialise an empty Chimera Widget instance'''
        super(ChimeraWidget, self).__init__(parent)
        
        # parameters
        self.adj = {}       # (m,n,h,l) keyed adjacency dict of chimera structure
        
        self.clicked_tile = None    # stored (m,n)pair for a clicked tile
        self.active_range = None    # range of m and n for selected region
        
        self.mouse_pos = None
        self.initUI()
        
    def initUI(self):
        '''Initialise the UI'''
        
        self.tile_layout = QtGui.QGridLayout(self)
        self.canvas - Canvas(self)
        self.setWidget(self.canvas)
        self.canvas.setLayout(self.tile_layout)
    
    def set_color_map(self, typ, cmap):
        '''Set the colorring method for the specified sub-class'''
        
        if typ == 'node':
            for kt, tile in self.canvas.tiles.items():
                for kn, node in tile.nodes.items():
                    node.node_color_map = cmap

        elif typ == 'edge':
            self.canvas.edge_color_map = cmap

        elif typ == 'tile':
            for kt, tile in self.canvas.tiles.items():
                tile.tile_color_map = cmap

        else:
            print('Invalid sub-clas type')

    def set_color_pars(self, typ, pars):
        '''Set the color map parameters for each item in pars'''
        
        if typ == 'node':
            for kp, p in pars.items():
                kt, kn = kp[:2], kp[2:]
                try:
                    node = self.canvas.tiles[kt].nodes[kn]
                    node.node_color_par = p
                except KeyError:
                    print('Invalid node key set: {0},{1}'.format(kt, kn))
                
        elif typ == 'edge':
            self.canvas.edge_color_pars = pars
        elif typ == 'tile':
            for kt, p in pars.items():
                try:
                    tile = self.canvas.tiles[kt]
                    tile.tile_color_par = p
                except KeyError:
                    print('Invalid tile key: {0}'.format(kt))
        else:
            print('Invalid sub-clas type')
        
    def build_tile_grid(self, M, N, adj):
        '''Build the grid layout from the tile range and adjacency'''
        
        # forget tiles
        for key, tile in self.canvas.tiles.items():
            tile.setParent(None)
        self.canvas.tiles = {}
        
        # release grid widgets
        while self.tile_layout.count():
            self.tile_layout.takeAt(0).widget().deleteLater()
            
        # determine new canvas size
        width = N*TILE_SIZE
        height = M*TILE_SIZE
        
        # update Canvas size parameters
        self.canvas.w = width
        self.canvas.h = height
        self.canvas.scaling = 1.
        self.canvas.setGeometry(0, 0, width, height)
        
        # convert adjacency dict to tup format if index keyed, assume valid ind
        if type(adj.keys()[0]) is int:
            adj = {linear_to_tuple(k, M, N):\
                [linear_to_tuple(a, M, N) for a in adj[k]] for k in adj}
        
        self.adj = adj
            
    def load_from_file(self, fname):
        '''Load the chimera architecture from a formatted file'''
        
        try:
            M, N, adj = load_chimera_file(fname)
        except IOError:
            print('Failed to load given ffile...')
            return
        
        self.build_tile_grid(M, N, adj)
        
    
    def load_from_adj(self, adj, M=None, N=None):
        '''Construct the chimera architecture from an adj dict with tup keys'''
        
        # find range of m, n values
        if M is None or N is None:
            ms, ns = zip(*[k[:2] for k in adj])         # all m and n values
            M0, N0 = [min(x) for x in [ms, ns]]         # tile index offset
            M, N = [max(x)-min(x) for x in [ms, ns]]    # zeroed tile index span
        else:
            M0, N0  = 0, 0
        
        # offset adj tile indices
        adj_ = {}
        for k, a in adj.items():
            adj_[(k[0]-M0, k[1]-N0, k[2], k[3])] = a
        
        self.build_tile_grid(M, N, adj_)
    
    # virtual methods
    
    def on_node_click(self, tup):
        ''' '''
        pass
    
    # interrupts
    