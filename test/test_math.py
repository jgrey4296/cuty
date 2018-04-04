import unittest
import logging
import numpy as np
import IPython
from test_context import cairo_utils as utils
from cairo_utils import math as cumath
from cairo_utils.constants import IntersectEnum as IE
import math


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
        self.assertTrue(-1 <= circ.min() <= -0.99)
        self.assertTrue(0.99 <= circ.max() <= 1)


    #interpolate
    def test_interpolate(self):
        arr = np.array([[0,0],[0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1,1]])
        i_arr = cumath._interpolate(arr, 100)
        self.assertIsInstance(i_arr, np.ndarray)
        self.assertEqual(i_arr.shape, (100, 2))
        self.assertTrue((i_arr >= 0).all())
        self.assertTrue((i_arr <= 1).all())
        
    
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
        line = cumath.createLine(0,0, 0, 1, 10)
        self.assertTrue(len(line) == 10)
        self.assertTrue(line[:,0].min() == 0)
        self.assertTrue(line[:,0].max() == 0)
        self.assertTrue(line[:,1].min() == 0)
        self.assertTrue(line[:,1].max() == 1)
    
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
        a = np.array([1,0])
        result = cumath.rotatePoint(a, rads=math.radians(90))
        self.assertTrue(np.allclose(result, np.array([0,1])))

    
    #randomRad
    def test_randomRad(self):
        return
    
    #rotMatrix
    def test_rotMatrix(self):
        return
    
    #checksign
    def test_checksign(self):
        return
    
    #intersect
    def test_intersect(self):
        """ Check two simple lines intersect """
        l1 = np.array([0, 0, 0, 1]).reshape((2,2))
        l2 = np.array([-1,0.5, 1, 0.5]).reshape((2,2))
        intersection = cumath.intersect(l1, l2)
        self.assertIsNotNone(intersection)
        self.assertTrue((intersection == np.array([0, 0.5])).all())

    def test_intersect_fail(self):
        """ Check that two lines that intersect if infinite, don't intersect
        because of limited length """
        l1 = np.array([0, 0, 0, 1]).reshape((2,2))
        l2 = np.array([-2,0.5, -1, 0.5]).reshape((2,2))
        intersection = cumath.intersect(l1, l2)
        self.assertIsNone(intersection)

    def test_intersect_parallel_lines(self):
        """ Check that two parallel lines don't intersect """
        l1 = np.array([0, 0, 0, 1]).reshape((2,2))
        l2 = np.array([0.5, 0, 0.5, 1]).reshape((2,2))
        intersection = cumath.intersect(l1, l2)
        self.assertIsNone(intersection)

    def test_intersect_negative(self):
        l1 = np.array([-3, -2, -0, -2]).reshape((2,2))
        l2 = np.array([-2, -3, -2, -1]).reshape((2,2))
        intersection = cumath.intersect(l1, l2)
        self.assertIsNotNone(intersection)

        
    #is_point_on_line
    def test_is_point_on_line(self):
        """ Test whether points lie on a line or not """
        l1 = np.array([0, 0, 0, 1]).reshape((2,2))
        p1 = np.array([0, 0.5])
        self.assertTrue(cumath.is_point_on_line(p1, l1))

        l2 = np.array([0, 0, 1, 1]).reshape((2,2))
        p2 = np.array([0.5, 0.5])
        self.assertTrue(cumath.is_point_on_line(p2, l2))

        p3 = np.array([1.5, 1.5])
        self.assertFalse(cumath.is_point_on_line(p3, l2))
    
        p4 = np.array([-1.5, -1.5])
        self.assertFalse(cumath.is_point_on_line(p4, l2))
        
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

    def test_bbox_to_lines(self):
        bbox = np.array([1,2,3,4])
        result = cumath.bbox_to_lines(bbox, epsilon=0)
        lines = [l.flatten() for (l, e) in result]
        enums = [e for (l,e) in result]
        self.assertEqual(enums, [ IE.HBOTTOM, IE.HTOP, IE.VLEFT, IE.VRIGHT ])
        self.assertTrue((lines[0] == np.array([1,2,3,2])).all())
        self.assertTrue((lines[1] == np.array([1,4,3,4])).all())
        self.assertTrue((lines[2] == np.array([1,2,1,4])).all())
        self.assertTrue((lines[3] == np.array([3,2,3,4])).all())
                        
        
    
          
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
