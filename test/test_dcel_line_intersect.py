import unittest
import logging
import IPython
import numpy as np
from random import shuffle
from math import radians
from test_context import cairo_utils as utils
from cairo_utils import dcel
from cairo_utils.math import get_distance_raw
import cairo_utils.dcel.line_intersector as li
from cairo_utils.rbtree import Directions

class MockNode:
    def __init__(self, a):
        self.value = a

class DCEL_LINE_INTERSECT_Tests(unittest.TestCase):
    def setUp(self):
        self.dc = dcel.DCEL()

    def tearDown(self):
        self.dc = None

    #----------
    def test_comparison_function_simple_right(self):
        e1 = self.dc.create_edge(np.array([0,1]), np.array([0,0]))
        e2 = self.dc.create_edge(np.array([1,1]), np.array([1,0]))
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0.5}), Directions.RIGHT)
    
    def test_comparison_function_simple_left(self):
        e1 = self.dc.create_edge(np.array([0,1]), np.array([0,0]))
        e2 = self.dc.create_edge(np.array([1,1]), np.array([1,0]))
        self.assertEqual(li.line_cmp(MockNode(e2), e1, {"nudge": 1e-2, "y": 0.5}), Directions.LEFT)

    def test_comparison_function_simple_defaults_to_right(self):
        e1 = self.dc.create_edge(np.array([1,1]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([1,1]), np.array([1,0]))
        self.assertEqual(li.line_cmp(MockNode(e2), e1, {"nudge": 1e-2, "y": 0.5}), Directions.RIGHT)

    def test_comparison_function_crossover_right(self):
        e1 = self.dc.create_edge(np.array([0,1]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([1,1]), np.array([0,0]))
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 1}), Directions.RIGHT)

    def test_comparison_function_crossover_left(self):
        e1 = self.dc.create_edge(np.array([0,1]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([1,1]), np.array([0,0]))
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0}), Directions.LEFT)
        
    def test_comparison_function_crossover_defaults_to_left(self):
        e1 = self.dc.create_edge(np.array([0,1]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([1,1]), np.array([0,0]))
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {'nudge':1e-2, "y": 0.5}), Directions.RIGHT)

    def test_comparison_function_flat_simple_right(self):
        e1 = self.dc.create_edge(np.array([0.5,1]), np.array([0.5,0]))
        e2 = self.dc.create_edge(np.array([0,0.5]), np.array([1,0.5]))
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0.5, "x": 0.6}), Directions.RIGHT)
        
    def test_comparison_function_flat_simple_left(self):
        e1 = self.dc.create_edge(np.array([0.5,1]), np.array([0.5,0]))
        e2 = self.dc.create_edge(np.array([0,0.5]), np.array([1,0.5]))
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0.5, "x": 0.4}), Directions.LEFT)
        
    def test_comparison_function_flat_simple_defaults_to_left(self):
        e1 = self.dc.create_edge(np.array([0.5,1]), np.array([0.5,0]))
        e2 = self.dc.create_edge(np.array([0,0.5]), np.array([1,0.5]))
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0.5, "x": 0.5}), Directions.RIGHT)

    def test_comparison_function_double_flat_out_of_bounds(self):
        e1 = self.dc.create_edge(np.array([0.5,0]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([0.2,0]), np.array([0.7,0]))
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0, "x": 0}), Directions.LEFT)
    
    def test_comparison_function_double_flat_in_bounds(self):
        e1 = self.dc.create_edge(np.array([0.5,0]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([0.2,0]), np.array([0.7,0]))
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0, "x": 0.2}), Directions.LEFT)

    def test_comparison_function_double_flat_right(self):
        e1 = self.dc.create_edge(np.array([0.5,0]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([0.7, 0]), np.array([1.2,0]))
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y":0, "x":1.1}), Directions.RIGHT)
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y":0, "x":1}), Directions.RIGHT)
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y":0, "x":0.8}), Directions.RIGHT)

    def test_comparison_function_double_flat_dual(self):
        e1 = self.dc.create_edge(np.array([0.5,0]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([0.7, 0]), np.array([1.2,0]))
        #reminder: can two hlines have one to the right?
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y":0, "x":1.1}), Directions.RIGHT)
        
    def test_comparison_function_double_flat_left(self):
        e1 = self.dc.create_edge(np.array([0.5,0]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([0.2,0]), np.array([0.7,0]))
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0, "x": 0.3}), Directions.LEFT)

    def test_comparison_function_double_flat_defaults_to_right(self):
        e1 = self.dc.create_edge(np.array([0.5,0]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([0.2,0]), np.array([0.7,0]))
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0, "x": 0.5}), Directions.RIGHT)

    def test_comparison_function_ends_touch_left(self):
        e1 = self.dc.create_edge(np.array([0,1]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([0,0.5]), np.array([1,0]))

        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0.5}), Directions.LEFT)
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0}), Directions.LEFT)

    def test_comparison_function_ends_touch_right(self):
        e1 = self.dc.create_edge(np.array([1,1]), np.array([0,0]))
        e2 = self.dc.create_edge(np.array([1,0.5]), np.array([0,0]))

        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0.5}), Directions.RIGHT)
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0}), Directions.RIGHT)

    def test_comparison_function_flat_vert_ends_touch(self):
        e1 = self.dc.create_edge(np.array([1,1]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([0,0]), np.array([1,0]))
        
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0, 'x': 1}), Directions.RIGHT)
        self.assertEqual(li.line_cmp(MockNode(e1), e2, {"nudge": 1e-2, "y": 0, 'x': 0}), Directions.LEFT)

    def test_comparison_function_flat_as_itself(self):
        e1 = self.dc.create_edge(np.array([0,0]), np.array([1,0]))
        self.assertEqual(li.line_cmp(MockNode(e1), e1, {"nudge": 1e-2, "y": 0, "x":0.5}), Directions.RIGHT)

        
    #TODO: test intersections with added deltas, start points.
    #TODO: horizontal end points touching a line middle
        
    
    def test_intersect_halfedges_simple(self):
        logging.debug("Test Intersect HalfEdges Simple")
        #create some edges
        e1 = self.dc.create_edge(np.array([0,0]),np.array([1,0]))
        e2 = self.dc.create_edge(np.array([1,0]),np.array([0,1]))
        e3 = self.dc.create_edge(np.array([0.5,0.9]), np.array([0.5,-0.9]))
        #intersect
        results = self.dc.intersect_half_edges()
        self.assertTrue(len(results) == 3)
        #first intersection
        i1 = [x for x in results if np.allclose(x.vertex.loc, np.array([0.5,0.5]))][0]
        self.assertIsNotNone(i1)
        self.assertTrue(e2.twin in i1.contain)
        self.assertTrue(e3 in i1.contain)
        #second intersection:
        i2 = [x for x in results if np.allclose(x.vertex.loc, np.array([1,0]))][0]
        self.assertIsNotNone(i2)
        self.assertTrue(e1 in i2.end)
        self.assertTrue(e2.twin in i2.end)
        #Start-end match:
        i3 = [x for x in results if np.allclose(x.vertex.loc, np.array([0.5,0]))][0]
        self.assertTrue(e1 in i3.contain)
        self.assertTrue(e3 in i3.contain)

    def test_intersect_halfedges_non_flat(self):
        logging.debug("Test Intersect HalfEdges no flat")
        #create some edges
        e1 = self.dc.create_edge(np.array([0,0]),np.array([1,0.5]))
        e2 = self.dc.create_edge(np.array([1,0.5]),np.array([0,1]))
        e3 = self.dc.create_edge(np.array([0.5,1]), np.array([0.5,-1]))
        #intersect
        results = self.dc.intersect_half_edges()
        self.assertTrue(len(results) == 3)
        #first intersection
        i1 = [x for x in results if np.allclose(x.vertex.loc, np.array([0.5,0.75]))][0]
        self.assertIsNotNone(i1)
        self.assertTrue(e2.twin in i1.contain)
        self.assertTrue(e3 in i1.contain)
        #second intersection:
        i2 = [x for x in results if np.allclose(x.vertex.loc, np.array([1,0.5]))][0]
        self.assertIsNotNone(i2)
        self.assertTrue(e1.twin in i2.start)
        self.assertTrue(e2.twin in i2.end)
        #Start-end match:
        i3 = [x for x in results if np.allclose(x.vertex.loc, np.array([0.5,0.25]))][0]
        self.assertTrue(e1.twin in i3.contain)
        self.assertTrue(e3 in i3.contain)
        
    def test_intersect_halfedges_no_intersections(self):
        logging.debug("Test Intersect HalfEdges no intersections")
        #create
        e1 = self.dc.create_edge(np.array([0,0]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([0,1]), np.array([1,2]))
        e3 = self.dc.create_edge(np.array([-2,0]), np.array([0,4]))
        #intersect
        results = self.dc.intersect_half_edges()
        #verify
        self.assertEqual(len(results), 0)
        
    def test_intersect_halfedges_endpoints(self):
        logging.debug("Test Intersect HalfEdges endpoints")
        e1 = self.dc.create_edge(np.array([0,0]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([1,0]), np.array([1,1]))
        e3 = self.dc.create_edge(np.array([1,1]), np.array([0,0]))
        results = self.dc.intersect_half_edges()
        self.assertEqual(len(results), 3)

    def test_intersect_three_vert_one_horizontal(self):
        logging.debug("Test Intersect HalfEdges horizontal")
        v1 = self.dc.create_edge(np.array([0,0]), np.array([0,2000]))
        v2 = self.dc.create_edge(np.array([1000,0]), np.array([1000,2000]))
        v3 = self.dc.create_edge(np.array([2000,0]), np.array([2000,2000]))
        
        h1 = self.dc.create_edge(np.array([0,2000]), np.array([2000,2000]))
        results = self.dc.intersect_half_edges(edge_set=set([v1,v2,v3,h1]))
        self.assertEqual(len(results), 3)
        
        
        
if __name__ == "__main__":
      #use python $filename to use this logging setup
      LOGLEVEL = logging.DEBUG
      logFileName = "log.DCEL_Line_intersect_Tests"
      logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
      console = logging.StreamHandler()
      console.setLevel(logging.WARN)
      logging.getLogger().addHandler(console)
      unittest.main()
      #reminder: user logging.getLogger().setLevel(logging.NOTSET) for log control
