""" Constants used throughout the cairo_utils
    Mainly some colours, and math shortcuts
"""
from enum import Enum
from math import pi
import logging as root_logger
import sys
import numpy as np


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
EPSILON = max(1e-12, sys.float_info.epsilon)
D_EPSILON = 1e-6 #EPSILON * 10000000
TOLERANCE = 0.5
ALLCLOSE_TOLERANCE = [1e-10, 1e-10]

DELTA = 1 / 100
HALFDELTA = DELTA * 0.5
NODE_NUM = 100
NODE_RECIPROCAL = 1 / NODE_NUM

VERTRAD = 10

#Colours
TEXT = np.array([0, 1, 1, 1])
EDGE = np.array([1, 0, 0, 1])
VERTEX = np.array([1, 0, 1, 1])
FACE = np.array([0, 0, 1, 1])
START = np.array([0, 1, 0, 1])
END = np.array([1, 0, 0, 1])
BACKGROUND = np.array([0, 0, 0, 1])
FRONT = np.array([0.8, 0.1, 0.71, ALPHA])

IntersectEnum = Enum("BBox Intersect Edge", "VLEFT VRIGHT HTOP HBOTTOM")
