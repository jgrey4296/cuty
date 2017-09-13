""" Line: Representation of a 2d line """
from math import sqrt
from numbers import Number
import logging as root_logger
import numpy as np
import cairo_utils as utils

logging = root_logger.getLogger(__name__)

CENTRE = np.array([[0.5, 0.5]])

class Line:
    """ A line as a start x and y, a direction, and a length """

    def __init__(self, sx, sy, dx, dy, l, swapped=False):
        assert(all([isinstance(x, Number) for x in [sx, sy, dx, dy, l]]))
        self.source = np.array([sx, sy])
        self.direction = np.array([dx, dy])
        self.length = l
        self.swapped = swapped

    def constrain(self, min_x, min_y, max_x, max_y):
        """ Intersect the line with a bounding box, adjusting points as necessary """
        #min and max: [x,y]
        dest = self.destination()
        npArray_line = np.array([*self.source, *dest])
        bbox_lines = [np.array([min_x, min_y, max_x, min_y]),
                      np.array([min_x, max_y, max_x, max_y]),
                      np.array([min_x, min_y, min_x, max_y]),
                      np.array([max_x, min_y, max_x, max_y])]
        #intersect one of the bbox lines
        p = None
        while p is None and bool(bbox_lines):
            p = utils.intersect(npArray_line, bbox_lines.pop())
        if p is not None:
            summed = pow(p[0]-self.source[0], 2) + pow(p[1]-self.source[1], 2)
            new_length = sqrt(summed)
            if new_length != 0:
                self.length = new_length
            else:
                logging.warning("Line: new calculated length is 0")

    def destination(self):
        """ Calculate the destination vector of the line """
        ex = self.source[0] + (self.length * self.direction[0])
        ey = self.source[1] + (self.length * self.direction[1])
        return np.array([ex, ey])

    def bounds(self):
        if self.swapped:
            return np.row_stack((self.destination(), self.source))
        else:
            return np.row_stack((self.source, self.destination()))

    @staticmethod
    def newLine(a, b, bbox):
        """ Create a new line from two vertices """
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)
        #Calculate the line parameters:
        swapped = False
        d_a = utils.get_distance(np.array([[a.x, a.y]]), CENTRE)
        d_b = utils.get_distance(np.array([[b.x, b.y]]), CENTRE)
        aInBBox = a.within(bbox)
        bInBBox = b.within(bbox)
        if d_b < d_a and bInBBox:
            logging.debug("Swapping vertices for line creation, source is now: {}".format(b))
            temp = a
            a = b
            b = temp
            swapped = True
        vx = b.x - a.x
        vy = b.y - a.y
        l = sqrt(pow(vx, 2) + pow(vy, 2))
        if l != 0:
            scale = 1/l
        else:
            scale = 0
        dx = vx * scale
        dy = vy * scale
        #cx = a.x + (dx * l)
        #cy = a.y + (dy * l)
        return Line(a.x, a.y, dx, dy, l, swapped=swapped)
