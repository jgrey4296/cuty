import unittest
import logging
import numpy as np
from test_context import cairo_utils as utils
from cairo_utils import dcel
import IPython
#https://ipython.readthedocs.io/en/stable/config/options/terminal.html
#IPython.embed(simple_prompt=True)
#in shell: ipython --simple-prompty --matplotlib

class DCEL_VERTEX_Tests(unittest.TestCase):
    def setUp(self):
        self.dc = dcel.DCEL()
        self.v = dcel.Vertex(np.array([0.2, 0.6]), data={"Test": 5})
        return 1

    def tearDown(self):
        self.dc = None
        self.v = None
        return 1

    #----------
    #test dcel creation
    def test_dcel_creation(self):
        self.assertIsInstance(self.dc, dcel.DCEL)
        
    #test Vertex
    def test_vertex_creation(self):
        self.assertIsInstance(self.v, dcel.Vertex)
        self.assertEqual(self.v.loc[0], 0.2)
        self.assertEqual(self.v.loc[1], 0.6)
        self.assertEqual(self.v.data["Test"], 5)

    def test_dcel_created_vertex(self):
        aDCVert = self.dc.newVertex(np.array([0.2, 0.6]), data={"Test": 5})
        self.assertIsInstance(aDCVert, dcel.Vertex)
        self.assertEqual(aDCVert.loc[0], 0.2)
        self.assertEqual(aDCVert.loc[1], 0.6)
        self.assertNotEqual(aDCVert.index, self.v.index)
        self.assertNotEqual(aDCVert, self.v)

    def test_vertex_export_import(self):
        exported = self.v._export()
        self.assertTrue(all([x in exported for x in ["i","x","y","halfEdges","data","active"]]))

    def test_vertex_activation(self):
        self.assertTrue(self.v.active)
        self.v.deactivate()
        self.assertFalse(self.v.active)
        self.v.activate()
        self.assertTrue(self.v.active)

    def test_vertex_edge_registration_and_edgeless(self):
        self.assertTrue(self.v.isEdgeless())
        he1 = dcel.HalfEdge()
        he2 = dcel.HalfEdge()
        self.v.registerHalfEdge(he1)
        self.assertFalse(self.v.isEdgeless())
        self.v.registerHalfEdge(he2)
        self.assertFalse(self.v.isEdgeless())
        self.v.unregisterHalfEdge(he2)
        self.assertFalse(self.v.isEdgeless())
        self.v.unregisterHalfEdge(he1)
        self.assertTrue(self.v.isEdgeless())

    def test_vertex_bboxs(self):
        #check the bbox is created appropriately, from self and free
        #bbox
        bbox = self.v.bbox(e=1)
        self.assertIsInstance(bbox, np.ndarray)
        self.assertEqual(bbox.shape, (4,))
        self.assertTrue(all(bbox == np.array([-0.8, -0.4, 1.2, 1.6])))

        free_bbox = dcel.Vertex.free_bbox(np.array([0.2, 0.6]), e=1)
        self.assertTrue(all(free_bbox == bbox))
        
        #within_circle:
        self.assertTrue(self.v.within_circle(np.array([0,0]), 1.5))
        self.assertFalse(self.v.within_circle(np.array([0,0]), 0.2))
        self.assertFalse(self.v.within_circle(np.array([-2,-2]),1))
        self.assertTrue(self.v.within_circle(np.array([-2,-2]),4))
        
        #within:
        self.assertTrue(self.v.within(free_bbox))
        self.assertFalse(self.v.within( free_bbox - 5))
        #outside:
        self.assertFalse(self.v.outside(free_bbox))
        self.assertTrue(self.v.outside(free_bbox - 5))
        

    def test_vertex_get_nearby(self):
        #create multiple vertices, get all nearby
        v1 = self.dc.newVertex(np.array([0.2, 0.6]))
        self.dc.newVertex(np.array([0.3,0.4]))
        self.dc.newVertex(np.array([0.1,0.5]))
        nearby = v1.get_nearby_vertices(e=1)
        self.assertEqual(len(nearby), 3)
        self.dc.newVertex(np.array([0.3,0.35]))
        nearby2 = v1.get_nearby_vertices(e=1)
        self.assertEqual(len(nearby2), 4)

    def test_vertex_to_array(self):
        arr = self.v.toArray()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (2,))
        self.assertTrue(all(arr == np.array([0.2, 0.6])))
        
    def test_vertex_extend_to_line(self):
        #create a line appropriately
        v1 = self.dc.newVertex(np.array([0,0]))
        e = v1.extend_line_to(target=np.array([1,0]))
        self.assertIsNotNone(e)
        self.assertIsInstance(e, dcel.HalfEdge)
        self.assertTrue(all(e.twin.origin.toArray() == np.array([1,0])))
        self.assertTrue(all(e.origin.toArray() == np.array([0,0])))
        self.assertFalse(v1.isEdgeless())
        
    def test_vertex_get_sorted_edges(self):
        #create multiple lines from,
        #return in sorted order ccw
        v1 = self.dc.newVertex(np.array([0,0]))
        coords = np.array([[1,0],[0.5, 0.5], [0, 1],
                           [-0.5, 0.5], [-1,0], [-0.5, -0.5],
                           [0.5, -0.5]])
        for x in coords[:,]:
            v1.extend_line_to(target=x)

        sorted_edges = v1.get_sorted_edges()
        self.assertEqual(len(sorted_edges), len(coords))
        
        
        self.assertTrue(True)
        

if __name__ == "__main__":
      #use python $filename to use this logging setup
      LOGLEVEL = logging.INFO
      logFileName = "log.DCEL_VERTEX_Tests"
      logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
      console = logging.StreamHandler()
      console.setLevel(logging.WARN)
      logging.getLogger().addHandler(console)
      unittest.main()
      #reminder: user logging.getLogger().setLevel(logging.NOTSET) for log control
