""" Constants used throughout the cairo_utils 
    Mainly some colours, and math shortcuts
"""
from math import pi
from enum import Enum
import numpy as np
import logging as root_logger
import sys

logging = root_logger.getLogger(__name__)

#constants:
#BBOX = [min_x, min_y, max_x, max_y]
SMALL_RADIUS = 15
FONT_SIZE = 100
WIDTH = 10
ALPHA = 0.1
PI = pi
TWOPI = 2 * pi
QUARTERPI = 0.5 * pi
THREEFOURTHSTWOPI = 3/4 * TWOPI
EPSILON = sys.float_info.epsilon
D_EPSILON = EPSILON * 20
TOLERANCE = 0.5

VERTRAD = 10

TEXT = np.array([0, 1, 1, 1])
EDGE = np.array([1, 0, 0, 1])
VERTEX = np.array([1, 0, 1, 1])
FACE = np.array([0, 0, 1, 1])
START = np.array([0, 1, 0, 1])
END = np.array([1, 0, 0, 1])
BACKGROUND = np.array([0, 0, 0, 1])
FRONT = np.array([0.8, 0.1, 0.71, ALPHA])

IntersectEnum = Enum("BBox Intersect Edge", "VLEFT VRIGHT HTOP HBOTTOM")
