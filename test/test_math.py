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
    def test_interpolate(self):
        arr = np.array([[0,0], [1,1]])
        i_arr = cumath._interpolate(arr, 100)
        self.assertIsInstance(i_arr, np.ndarray)
        self.assertEqual(i_arr.shape, (100, 2))
        self.assertTrue(all(i_arr >= 0))
        self.assertTrue(all(i_arr <= 1))
        
    
    #gerDirections
    def test_getDirections(self):
        return                           

    #granulate
    def test_granulate(self):
        return
    
    #vary
    def test_vary(self):
        return
    
    #sampleAlongLine
    def test_sampleAlongLine(self):
        return

    #createLine
    def test_createLine(self):
        return
    
    #bezier1cp
    def test_bezier1cp(self):
        return
    
    #bezier2cp
    def test_bezier2cp(self):
        return
    
    def test_get_distances(self):
        return
    
    #get_normal
    def test_get_normal(self):
        return

    #get_bisector
    def test_get_bisector(self):
        return
    
    #get_circle_3p
    def test_get_circle_3p(self):
        return
    
    #extend_line
    def test_extend_line(self):
        return
    
    #get_midpoint
    def test_get_midpoint(self):
        return
    
    #rotate_point
    def test_rotatePoint(self):
        return
    
    #randomRad
    def test_randomRad(self):
        return
    
    #rotMatrix
    def test_rotMatrix(self):
        return
    
    #checksign
    def test_checksign(a, b):
        return
    
    #intersect
    def test_intersect(self):
        return
    
    #random_points
    def test_random_points(self):
        return
    
    #bound_line_in_bbox
    def test_bound_line_in_bbox(self):
        return
    
    #makeHorizontalLine
    def test_makeHorizontalLine(self):
        return
    
    #makeVerticalLine
    def test_makeVerticalLine(self):
        return
    
    #get_lowest_point_on_circle
    def test_get_lowest_point_on_circle(self):
        return
    
    #sort_coords
    def test_sort_coords(self):
        return
    
    #inCircle
    def test_inCircle(self):
        return
    
    #isClockwise
    def test_isClockwise(self):
        return
    
    #getMinRangePair
    def test_getMinRangePair(self):
        return
    
    #getClosestToFocus
    def test_getClosestToFocus(self):
        return
    
    #get_closests_on_side
    def test_get_closest_on_side(self):
        return
    
    #angle_between_points
    def test_get_angle_between_points(self):
        return
    
    #displace_along_line
    def test_displace_along_line(self):
        return
    
    #clamp
    def test_clamp(self):
        return
    
          
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
