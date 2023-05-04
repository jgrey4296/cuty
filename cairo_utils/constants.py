""" Constants used throughout the cairo_utils
    Mainly some colours, and math shortcuts
"""
##-- imports
from __future__ import annotations
from enum import Enum
from math import pi
from typing import Final
import logging as root_logger
import sys
import numpy as np

##-- end imports

logging = root_logger.getLogger(__name__)

#BBOX                      = [min_x, min_y, max_x, max_y]
SMALL_RADIUS       : Final =  15
FONT_SIZE          : Final =  100
WIDTH              : Final =  10
ALPHA              : Final =  0.1
PI                 : Final =  pi
TWOPI              : Final =  2 * pi
QUARTERPI          : Final =  0.5 * pi
THREEFOURTHSTWOPI  : Final =  3/4 * TWOPI
EPSILON            : Final =  max(1e-12, sys.float_info.epsilon)
D_EPSILON          : Final =  1e-6 #EPSILON * 10000000
TOLERANCE          : Final =  0.5
ALLCLOSE_TOLERANCE : Final =  [1e-10, 1e-10]

DELTA              : Final =  1 / 100
HALFDELTA          : Final =  DELTA * 0.5
NODE_NUM           : Final =  100
NODE_RECIPROCAL    : Final =  1 / NODE_NUM

VERTRAD            : Final =  10

IntersectEnum      : Final =  Enum("BBox Intersect Edge", "VLEFT VRIGHT HTOP HBOTTOM")

TEXT              = [0, 1, 1, 1]
EDGE              = [1, 0, 0, 1]
VERTEX            = [1, 0, 1, 1]
FACE              = [0, 0, 1, 1]
START             = [0, 1, 0, 1]
END               = [1, 0, 0, 1]
BACKGROUND        = [0, 0, 0, 1]
FRONT             = [0.8, 0.1, 0.71, 0.5]
FACE_COLOUR       = [0.2, 0.2, 0.9, 1]
EDGE_COLOUR       = [0.4, 0.8, 0.1, 1]
VERT_COLOUR       = [0.9, 0.1, 0.1, 1]
BACKGROUND_COLOUR = [0, 0, 0, 1]
