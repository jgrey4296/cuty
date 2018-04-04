import unittest
import logging
import IPython
import numpy as np
from test_context import cairo_utils as utils
from cairo_utils import dcel
from math import radians


class DCEL_HALFEDGE_Tests(unittest.TestCase):
    def setUp(self):
        self.dc = dcel.DCEL()
        self.v1 = dcel.Vertex(np.array([0,0]))
        self.he = dcel.HalfEdge(origin=self.v1)
        self.e = self.dc.createEdge(np.array([0,0]), np.array([1,0]))
        return 1

    def tearDown(self):
        self.dc = None
        return 1

    #----------
    def test_halfedge_creation(self):
        self.assertIsNotNone(self.he)
        self.assertIsNotNone(self.e)
        self.assertIsInstance(self.he, dcel.HalfEdge)
        self.assertIsInstance(self.e, dcel.HalfEdge)

    
    def test_export_import(self):
        exported = self.e._export()
        self.assertTrue(all([x in exported for x in ["i","origin","twin","face","next","prev","data"]]))

    def test_split(self):
        originalEnd = self.e.twin.origin
        otherEnd = self.e.twin
        (newPoint, newEdge) = self.e.split(np.array([0.5, 0]))
        self.assertTrue(all(self.e.twin.origin.toArray() == np.array([0.5,0])))
        self.assertTrue(all(newEdge.origin.toArray() == np.array([0.5,0])))
        self.assertTrue(all(newEdge.twin.origin.toArray() == np.array([1,0])))

        self.assertTrue(newEdge.twin.origin == originalEnd)
        self.assertTrue(otherEnd.origin == newPoint)
        self.assertTrue(otherEnd.twin.origin == self.e.origin)
        
    def test_split_by_ratio(self):
        (newPoint, newEdge) = self.e.split_by_ratio(r=0.5)
        self.assertTrue(all(self.e.twin.origin.toArray() == np.array([0.5,0])))
        self.assertTrue(all(newEdge.origin.toArray() == np.array([0.5,0])))
        self.assertTrue(all(newEdge.twin.origin.toArray() == np.array([1,0])))

    def test_intersect(self):
        #create an intersecting edge
        e2 = self.dc.createEdge(np.array([0.5, 0.5]), np.array([0.5, -0.5]))
        intersection = self.e.intersect(e2)
        #check
        self.assertIsNotNone(intersection)
        self.assertIsInstance(intersection, np.ndarray)
        self.assertTrue(all(intersection == np.array([0.5, 0])))
        #check a none-intersection
        e3 = self.dc.createEdge(np.array([2,4]),np.array([3,4]))
        self.assertIsNone(self.e.intersect(e3))

    def test_intersects_bbox(self):
        #check intersection, non-intersection, and double intersection?
        #edge: 0,0 -> 1,0
        e = self.dc.createEdge(np.array([0,0]), np.array([1,0]))
        result = e.intersects_bbox(np.array([-0.5,-0.5,0.5,0.5]))
        self.assertEqual(len(result), 1)
        (coords, edgeI) = result[0]
        self.assertIsInstance(edgeI, utils.constants.IntersectEnum)
        self.assertEqual(edgeI, utils.constants.IntersectEnum.VRIGHT)

    #todo: add in a double edge bbox intersection test, and a null test

    def test_within(self):
        self.assertTrue(self.e.within(np.array([-0.5, -0.5, 1.5, 1.5])))
        self.assertTrue(self.e.outside(np.array([1.5, 1.5, 2,2])))
        self.assertFalse(self.e.outside(np.array([-0.5, -0.5, 1.5, 1.5])))
        self.assertFalse(self.e.within(np.array([1.5, .5, 2, 2])))
        
    def test_within_circle(self):
        self.assertTrue(self.e.within_circle(np.array([0,0]), 2))
        self.assertFalse(self.e.within_circle(np.array([0,0]), 0.2))
        self.assertTrue(self.e.within_circle(np.array([1,1]), 2))
                         
    def test_constrain(self):
        #edge is [[0,0], [1,0]]
        result = self.e.to_constrained(np.array([0,0,0.5,1]))
        self.assertIsNotNone(result)
        #check constraint is correct
        self.assertTrue(all(result.flatten() == np.array([0,0,0.5,0])))
        #check the original line is unmodified:
        self.assertFalse(all(self.e.toArray().flatten() == np.array([0,0,0.5,0])))

    def test_constrain_unaffected(self):
        #create an edge within the bbox
        edgeAsArray = self.e.toArray().flatten()
        result = self.e.to_constrained(np.array([0,0,1,1]))
        self.assertIsNotNone(result)
        #check it is unmodified
        self.assertTrue(all(self.e.toArray().flatten() == edgeAsArray))
        self.assertTrue(all(self.e.toArray().flatten() == result.flatten()))
        
    def test_connections_align_succeed(self):
        self.assertTrue(self.e.connections_align(self.e.twin))
        with self.assertRaises(Exception):
            self.e.connections_align(self.he)

    def test_vertices(self):
        #add vertex, clear vertices
        he = dcel.HalfEdge()
        e = dcel.HalfEdge(twin=he)
        v1 = dcel.Vertex(np.array([0,0]))
        v2 = dcel.Vertex(np.array([1,1]))
        self.assertIsNotNone(e.twin)
        self.assertIsNone(e.origin)
        self.assertIsNone(e.twin.origin)

        e.addVertex(v1)
        self.assertIsNotNone(e.origin)
        e.addVertex(v2)
        self.assertIsNotNone(e.twin.origin)
        with self.assertRaises(Exception):
            e.addVertex(v1)

        #check getVertices:
        tv1, tv2 = e.getVertices()
        self.assertTrue(tv1 == v1)
        self.assertTrue(tv2 == v2)        
            
        #check the vertices have registered the halfedges:
        self.assertTrue(e in v1.halfEdges)
        self.assertTrue(he in v2.halfEdges)
        #then clear and check the deregistration:
        e.clearVertices()
        self.assertIsNone(e.origin)
        self.assertIsNone(e.twin.origin)
        self.assertTrue(e not in v1.halfEdges)
        self.assertTrue(he not in v2.halfEdges)


    def test_faceSwap(self):
        #assign faces to each he, swap them
        f1 = dcel.Face()
        f2 = dcel.Face(site=np.array([1,1]))
        he = dcel.HalfEdge()
        e = dcel.HalfEdge(twin=he)
        e.face = f1
        he.face = f2
        self.assertEqual(e.face, f1)
        self.assertEqual(e.twin.face, f2)

        e.swapFaces()

        self.assertEqual(e.face, f2)
        self.assertEqual(e.twin.face, f1)
        

    def test_ordering(self):
        #test left turn
        centre = np.array([0,0])
        a = np.array([1,0])
        b = np.array([0,1])
        self.assertTrue(dcel.HalfEdge.ccw(centre, a, b))
        

    def test_vertex_retrieval(self):
        #get vertices as a tuple, and the raw coordinates as an array
        e = self.dc.createEdge(np.array([1,0]), np.array([0,1]))
        vs = e.getVertices()
        self.assertIsInstance(vs, tuple)
        self.assertTrue(all([isinstance(x, dcel.Vertex) for x in vs]))
        
        asArray = e.toArray()
        self.assertIsInstance(asArray, np.ndarray)
        self.assertEqual(asArray.shape, (2,2))

    def test_mark_for_cleanup(self):
        #Cleanup when marked test
        e = self.dc.createEdge(np.array([0,0]), np.array([1,1]))
        self.assertFalse(e.markedForCleanup)
        e.markForCleanup()
        self.assertTrue(e.markedForCleanup)
        self.assertFalse(e.twin.markedForCleanup)

    def test_is_infinite(self):
        self.assertTrue(self.he.isInfinite())
        self.assertFalse(self.e.isInfinite())

    def test_get_closer_and_further(self):
        e1 = self.dc.createEdge(np.array([1,0]), np.array([2,0]))
        res, switched = e1.getCloserAndFurther(np.array([0,0]))
        self.assertTrue((res == np.array([[1,0],[2,0]])).all())
        self.assertFalse(switched)
        
    def test_closer_and_further_switched(self):
        e1 = self.dc.createEdge(np.array([2,0]), np.array([1,0]))
        res, switched = e1.getCloserAndFurther(np.array([0,0]))
        self.assertTrue((res == np.array([[1,0],[2,0]])).all())
        self.assertTrue(switched)

    def test_closer_and_further2(self):
        e1 = self.dc.createEdge(np.array([1,0]), np.array([2,0]))
        res, switched = e1.getCloserAndFurther(np.array([3,0]))
        self.assertTrue((res == np.array([[2,0],[1,0]])).all())
        self.assertTrue(switched)

    def test_closer_and_further2(self):
        e1 = self.dc.createEdge(np.array([100,50]), np.array([200,50]))
        res, switched = e1.getCloserAndFurther(np.array([155,50]))
        self.assertTrue((res == np.array([[200,50],[100,50]])).all())
        self.assertTrue(switched)

    def test_rotate(self):
        #rotate the edge around a point
        e = self.dc.createEdge(np.array([0,0]), np.array([1,0]))
        result = e.rotate(np.array([0,0]), radians(90))
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2,2))
        self.assertTrue(np.allclose(result, np.array([[0,0], [0,1]])))

    def test_extend(self):
        e = self.dc.createEdge(np.array([0,0]), np.array([1,0]))
        e2 = e.extend(target=np.array([2,0]))
        self.assertTrue(e2.prev == e)
        self.assertTrue(e.next == e2)

        

    def test_avg_direction(self):
        return 0

    def test_bboxes(self):
        return 0

    def test_point_is_on_line(self):
        self.assertTrue(self.e.point_is_on_line(np.array([0.5,0])))

    def test_point_is_on_line_fail(self):
        self.assertFalse(self.e.point_is_on_line(np.array([2,0])))

    def test_ccw(self):
        a = np.array([0,0])
        b = np.array([1,0])
        c = np.array([0,1])
        self.assertTrue(dcel.HalfEdge.ccw(a,b,c))

    def test_ccw_fail(self):
        a = np.array([0,0])
        b = np.array([0,1])
        c = np.array([1,0])
        self.assertFalse(dcel.HalfEdge.ccw(a,b,c))

        
    def test_fix_faces(self):
        #create the halfedge
        #attach faces with dummy centroids
        #fix faces

        return 0
        

    
if __name__ == "__main__":
      #use python $filename to use this logging setup
      LOGLEVEL = logging.INFO
      logFileName = "log.DCEL_HALFEDGE_Tests"
      logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
      console = logging.StreamHandler()
      console.setLevel(logging.WARN)
      logging.getLogger().addHandler(console)
      unittest.main()
      #reminder: user logging.getLogger().setLevel(logging.NOTSET) for log control
