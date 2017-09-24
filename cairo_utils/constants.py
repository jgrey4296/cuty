from math import pi
import logging as root_logger
import sys

logging = root_logger.getLogger(__name__)

#constants:
TEXT = [0, 1, 1, 1]
EDGE = [1, 0, 0, 1]
VERTEX = [1, 0, 1, 1]
FACE = [0, 0, 1, 1]
START = [0, 1, 0, 1]
END = [1, 0, 0, 1]

SMALL_RADIUS = 0.003
FONT_SIZE = 0.03

ALPHA = 0.1
BACKGROUND = [0, 0, 0, 1]
FRONT = [0.8, 0.1, 0.71, ALPHA]
TWOPI = 2 * pi
QUARTERPI = 0.5 * pi
THREEFOURTHSTWOPI = 3/4 * TWOPI
EPSILON = sys.float_info.epsilon
