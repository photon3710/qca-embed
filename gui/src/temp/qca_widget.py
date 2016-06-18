from PyQt4 import QtGui, QtCore, QtSvg

from core.parse_qca import parse_qca_file
from core.auxil import convert_to_full_adjacency, convert_to_lim_adjacency,\
    prepare_convert_adj, CELL_FUNCTIONS

import os
import numpy as np

# QCACell drawing parameters
CELL_SEP = 70               # Pixels between QCA cells
CELL_SIZE = .8*CELL_SEP     # Size of QCA cell
DOT_RAD = 0.25*CELL_SIZE    # Radius of quantum dots
CANVAS_OFFSET = 0.3

# Pen parameters
CELL_PEN_WIDTH = max(1, int(0.05*CELL_SIZE))    #
CELL_PEN_COLOR = QtGui.QColor(180, 180, 180)    #
TEXT_PEN_WIDTH = max(1, int(0.05*CELL_SIZE))    #
TEXT_PEN_COLOR = QtGui.QColor(0, 0, 0)          #
INT_PEN_STYLE = {'strong': QtCore.Qt.SolidLine,
                 'weak': QtCore.Qt.DashLine}
INT_PEN_WIDTH = 3

# Magnification
MAX_MAG = 5             # maximum magnification
MIN_MAG = 0.1           # minimum magnification
MAG_STEP = 0.1          # magnification step
MAG_WHEEL_FACT = 0.2    # factor for wheel zoom

DRAG_FACT = 1.          # factor for drag response

# Basic colors
BLACK = QtGui.QColor(0, 0, 0, 255)
RED = QtGui.QColor(255, 0, 0)
BLUE = QtGui.QColor(0, 0, 255)

QCA_COL = {'default': QtGui.QColor(255, 255, 255),
           'inactive': QtGui.QColor(100, 100, 100),
           'output': QtGui.QColor(255, 255, 0),
           'input': QtGui.QColor(0, 200, 0),
           'fixed': QtGui.QColor(255, 165, 0),
           'clicked': QtGui.QColor(0, 150, 150)}


class QCACellWidget(QtGui.QWidget):
    '''Class describing a single QCA cell'''
    
    def __init__(self, canvas, cell, spacing=1., offset=[0, 0]):
        '''Initialise an QCA cell from a cell dict'''
        super(QCACellWidget, self).__init__(canvas)
        
        assert isinstance(canvas, Canvas), 'Must provide Canvas'

        self.canvas = canvas
        self.qca_widget = canvas.qca_widget

        # cell parameters
        self.x = CELL_SEP*(cell['x']-offset[0])*1./spacing
        self.y = CELL_SEP*(cell['y']-offset[1])*1./spacing

        self.cell_pen = QtGui.QPen(CELL_PEN_COLOR)
        self.cell_pen.setWidth(CELL_PEN_WIDTH)

        self.text_pen = QtGui.QPen(TEXT_PEN_COLOR)
        self.text_pen.setWidth(TEXT_PEN_WIDTH)

        self.type = cell['cf']
        self.qdots = []
        self.num = cell['number']
        for qd in cell['qdots']:
            x = CELL_SEP*(qd['x']-offset[0])*1./spacing
            y = CELL_SEP*(qd['y']-offset[1])*1./spacing
            self.qdots.append([x, y])
        
        # flags for cell type (simplifies access later)
        self.fixed = self.type == CELL_FUNCTIONS['QCAD_CELL_FIXED']
        self.driver = self.type == CELL_FUNCTIONS['QCAD_CELL_INPUT']
        self.output = self.type == CELL_FUNCTIONS['QCAD_CELL_OUTPUT']
        self.normal = not (self.fixed or self.driver)
        
        if self.fixed:
            self.pol = cell['pol']
        else:
            self.pol = None

        self.setGeometry(0, 0, CELL_SIZE, CELL_SIZE)
        self.clicked = False
        self.mouse_pos = None
        
        self.label = str(self.num)
        
        # external color parameters
        self.color_map = None   # function which returns a QColor
        self.color_par = None   # dict of for color_map parameters
        
    def get_color(self):
        '''Determine the background color of the cell under the current
        behaviour'''
        
        if self.color_map is None:   # default coloring scheme
            if self.clicked:
                color = QCA_COL['clicked']
            else:
                if self.type == 0:
                    color = QCA_COL['default']
                elif self.type == 1:    # input
                    color = QCA_COL['input']
                elif self.type == 2:    # output
                    color = QCA_COL['output']
                elif self.type == 3:    # fixed
                    color = RED if self.pol > 0 else BLUE
                else:
                    print('Invalid cell type')
                    color = QCA_COL['default']
        else:
            color = self.color_map(self, **self.color_par)

        return color
        
    def set_color_map(self, func):
        '''Set the color_map method, unset with func=None'''
        self.color_map = func
        
    def set_color_par(self, par):
        '''Set color_map parameters'''
        self.color_par = par

    # PAINTING METHODS

    def draw_cell(self, painter):
        ''' '''
        painter.setPen(self.cell_pen)
        painter.setBrush(self.get_color())

        painter.drawRect(self.geometry())

        # write cell label
        painter.setPen(self.text_pen)
        painter.setFont(QtGui.QFont('Decorative', 10))
        painter.drawText(self.geometry(), QtCore.Qt.AlignCenter, self.label)
    
    def paintEvent(self, e):
        ''' '''
        painter = QtGui.QPainter()
        painter.begin(self)
        self.draw_cell(painter)
        painter.end()
        
    # INTERRUPTS

    def mousePressEvent(self, e):
        ''' '''
        self.mouse_pos = e.pos()
        self.parent().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        ''' '''
        if self.mouse_pos is not None:
            diff = e.pos()-self.mouse_pos
            if max(abs(diff.x()), abs(diff.y())) < CELL_SIZE:
                self.qca_widget.on_click(self.num)
        self.mouse_pos = None
        self.parent().mouseReleaseEvent(e)
    
    def mouseDoubleClickEvent(self, e):
        '''On double click, echo cell info'''
        print('Clicked cell {0}'.format(self.num))


class Canvas(QtGui.QWidget):
    '''Drawing canvas for QCA cells'''
    
    def __init__(self, qca_widget):
        '''Initialise Canvas widget. Must specific parent QCAWidget'''
        super(Canvas, self).__init__(qca_widget)
        
        assert isinstance(qca_widget, QCAWidget), 'Must give QCAWidget'

        # parameters
        self.scaling = 1.
        self.qca_widget = qca_widget    # used to avoid self.parent() calls
        
        self.w = 1.     # width for scaling=1.
        self.h = 1.     # height for scaling=1.
        
    def rescale(self, zoom=True, f=1.):
        '''Rescale the Canvas (with limits)'''
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
        
    def move_cells(self):
        '''Change QCACellWidget geometries'''
        
        for cell in self.qca_widget.cells:
            x = cell.x*self.scaling
            y = cell.y*self.scaling
            size = CELL_SIZE*self.scaling
            cell.setGeometry(x, y, size, size)
            
    # PAINT METHODS
    
    def draw_cells(self, painter):
        ''' '''
        for cell in self.qca_widget.cells:
            cell.draw_cell(painter)

    def draw_connections(self, painter):
        ''' '''
        J0 = self.qca_widget.J0

        # store cell centers
        X = []
        Y = []
        for cell in self.qca_widget.cells:
            geo = cell.geometry()
            X.append(geo.x()+.5*geo.width())
            Y.append(geo.y()+.5*geo.height())

        # draw all non-zero interactions
        pen = QtGui.QPen(BLACK)
        pen.setWidth(max(1, INT_PEN_WIDTH*self.scaling))
        for i in xrange(J0.shape[0]-1):
            for j in xrange(i+1, J0.shape[0]):
                if abs(J0[i, j]) > 0.:
                    pen.setStyle(INT_PEN_STYLE[
                        'strong' if abs(J0[i, j]) > 0.5 else 'weak'])
                    painter.setPen(pen)
                    painter.drawLine(X[i], Y[i], X[j], Y[j])

    def paint(self, painter):
        ''' '''
        self.move_cells()
        self.draw_connections(painter)
        self.draw_cells(painter)
        
    # INTERRUPTS
        
    def paintEvent(self, e):
        ''' '''
        painter = QtGui.QPainter()
        painter.begin(self)
        self.paint(painter)
        painter.end()
    

class QCAWidget(QtGui.QScrollArea):
    '''Base class for QCAWidget control'''
    
    def __init__(self, parent=None, fname=None):
        '''Initialise an empty QCAWidget object'''
        super(QCAWidget, self).__init__(parent)
        
        # parameters
        self.cells = []         # list of QCACellWidgets
        self.spacing = 1.       # cell-cell spacing value
        self.J = np.zeros(0)    # cell-cell interaction matrix
        self.J0 = self.J        # normalized J with only included interactions
        
        # mouse tracking
        self.mouse_pos = None
        
        # initialise UI
        self.initUI()
        
        if fname is not None:
            self.update_circuit(fname)

    def initUI(self):
        '''Initialise '''
        
        # create main canvas widget
        self.canvas = Canvas(self)
        self.canvas.setGeometry(0, 0, 0, 0)
        self.setWidget(self.canvas)
        
    def update_circuit(self, fname, full_adj):
        ''' '''
        
        try:
            cells, spacing, zones, J, feedback = \
                parse_qca_file(fname, one_zone=True)
        except:
            print('Failed to load QCA File...')
            return
        
        # save_relative filename for reference
        self.relpath = os.path.relpath(fname, os.getcwd())
        
        # forget old circuit
        for cell in self.cells:
            cell.setParent(None)
        self.cells = []
        
        # update J coefficients
        self.J = J
        
        # set up adjacency conversion variables
        Js, T, A, DX, DY = prepare_convert_adj(cells, spacing, J)
        self.convert_vars = {'Js': Js, 'T': T, 'A': A, 'DX': DX, 'DY': DY}
        
        # set adjacency type
        self.set_adjacency(full_adj, update=False)  # sets full_adj and J0
        
        # find span and offset of circuit: currently inefficient
        x_min = min([cell['x'] for cell in cells])
        x_max = max([cell['x'] for cell in cells])
        y_min = min([cell['y'] for cell in cells])
        y_max = max([cell['y'] for cell in cells])
        
        o = CANVAS_OFFSET
        span = [spacing+x_max-x_min, spacing+y_max-y_min]
        offset = [x_min-o*span[0], y_min-o*span[1]]
        
        # update circuit constants
        self.spacing = spacing
        self.offset = offset

        # update size and scaling of canvas
        factor = (1+2*o)*CELL_SEP*1./spacing
        self.canvas.scaling = 1.
        self.canvas.w = span[0]*factor
        self.canvas.h = span[1]*factor
        self.canvas.setGeometry(0, 0,
                                self.canvas.w, self.canvas.h)

        # add new cells
        for c in cells:
            cell = QCACellWidget(self.canvas, c, 
                              spacing=self.spacing, offset=self.offset)
            self.cells.append(cell)
            cell.show()

        self.canvas.update()
        
    def set_adjacency(self, full_adj, update=True):
        ''' '''
        self.full_adj = full_adj
        if full_adj:
            self.J0 = convert_to_full_adjacency(self.J, **self.convert_vars)
        else:
            self.J0 = convert_to_lim_adjacency(self.J, **self.convert_vars)
        
        # only care about normalised version
        self.J0 /= np.max(np.abs(self.J0))
        
        if update:
            self.canvas.update()
    
    def select_cell(self, num):
        ''' '''
        try:
            cell = self.cells[num]
        except KeyError:
            print('Requested cell index invalid')
            return
        
        if cell.driver or cell.fixed or cell.clicked:
            return
        
        # unselect all cells
        for c in self.cells:
            if c.clicked:
                c.clicked=False
                self.canvas.update(c.geometry())
        
        # select requested cell
        cell.clicked = True
        self.canvas.update(cell.geometry())
        
    def save_svg(self, fname):
        '''Save the current QCA circuit canvas to an svg file'''
        
        generator = QtSvg.QSvgGenerator()
        generator.setFileName(fname)
        generator.setSize(self.canvas.size())
        generator.setViewBox(self.canvas.rect())
        
        painter = QtGui.QPainter()
        painter.begin(generator)
        self.canvas.paint(painter)
        painter.end()
        
    # behaviour modification
    
    def on_click(self, cell):
        '''virtual: action when cell clicked'''
        pass
    
    def set_color_map(self, func):
        '''Set the coloring behaviour for QCACells'''
        
        for cell in self.cells:
            cell.set_color_map(func)
            
    def set_color_pars(self, pars):
        '''Set the color_map parameters for each item in pars: pars[ind] = pars
        for cell[ind]'''
        
        for ind in pars:
            self.cells[ind].set_color_par(pars[ind])
        
    # interrupts
        
    def mousePressEvent(self, e):
        '''Cell clicking should be handled by virtual method on_click so
        only canvas dragging is implemented'''
        self.mouse_pos = e.pos()
        
    def mouseReleaseEvent(self, e):
        '''On mouse release, forget old mouse positions to avoid jumping'''
        self.mouse_pos = None
        
    def mouseMoveEvent(self, e):
        '''Pan canvas according to drag directions'''
        if self.mouse_pos is not None:
            diff = DRAG_FACT*(e.pos()-self.mouse_pos)
            self.mouse_pos = e.pos()
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value()-diff.y())
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value()-diff.x())
    
    def wheelEvent(self, e):
        '''Scrolling response'''
        
        mods = QtGui.QApplication.keyboardModifiers()   # binary-like flag
        
        if mods == QtCore.Qt.ControlModifier: # Ctrl pressed
            self.canvas.rescale(zoom=e.delta() > 0, f=MAG_WHEEL_FACT)
        else:
            # should allow default QScrollArea behaviour
            super(QCAWidget, self).wheelEvent(e)
            
    def keyPressEvent(self, e):
        ''' '''
        if e.key() == QtCore.Qt.Key_Minus:
            self.canvas.rescale(zoom=False)
        elif e.key() == QtCore.Qt.Key_Plus:
            self.canvas.rescale(zoom=True)
        self.parent().keyPressEvent(e)