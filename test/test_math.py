import unittest
import logging
import numpy as np
from test_context import cairo_utils as utils
from cairo_utils import math as cumath


class CUMath_Tests(unittest.TestCase):

    def setUp(self):
        return 1
    
    def tearDown(self):
        return 1

    #----------
    def test_sampleCircle(self):
        circ = cumath.sampleCircle(0, 0, 1, 1000)
        self.assertIsInstance(circ, np.ndarray)
        self.assertEqual(circ.shape, (1000, 2))
        self.assertTrue(np.isclose(circ.min(), -1))
        self.assertTrue(np.isclose(circ.max(), 1))

    #interpolate

    #gerDirections

    #granulate

    #vary

    #sampleAlongLine

    #createLine

    #bezier1cp

    #bezier2cp

    #get_distance_raw

    #get_distance

    #get_distance_xyxy

    #get_normal

    #get_bisector

    #get_circle_3p

    #extend_line

    #get_midpoint

    #rotate_point

    #randomRad

    #rotMatrix

    #checksign

    #intersect

    #random_points

    #bound_line_in_bbox

    #makeHorizontalLine

    #makeVerticalLine

    #get_lowest_point_on_circle

    #sort_coords

    #inCircle

    #isClockwise

    #getMinRangePair

    #getClosestToFocus

    #get_closests_on_side

    #angle_between_points

    #displace_along_line

    #clamp
          
          
if __name__ == "__main__":
      #use python $filename to use this logging setup
      LOGLEVEL = logging.INFO
      logFileName = "log.cu_math_tests"
      logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
      console = logging.StreamHandler()
      console.setLevel(logging.WARN)
      logging.getLogger().addHandler(console)
      unittest.main()
      #reminder: user logging.getLogger().setLevel(logging.NOTSET) for log control
