import unittest
import logging
import numpy as np
from math import radians
from test_context import cairo_utils as utils
from cairo_utils import dcel
from cairo_utils.dcel.constants import EditE
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
        """ Check the dcel exists """
        self.assertIsInstance(self.dc, dcel.DCEL)
        
    #test Vertex
    def test_vertex_creation(self):
        """ Check the Simple Vertex can be created """
        self.assertIsInstance(self.v, dcel.Vertex)
        self.assertEqual(self.v.loc[0], 0.2)
        self.assertEqual(self.v.loc[1], 0.6)
        self.assertEqual(self.v.data["Test"], 5)

    def test_dcel_created_vertex(self):
        """ Create a vertex through the dcel """
        aDCVert = self.dc.newVertex(np.array([0.2, 0.6]), data={"Test": 5})
        self.assertIsInstance(aDCVert, dcel.Vertex)
        self.assertEqual(aDCVert.loc[0], 0.2)
        self.assertEqual(aDCVert.loc[1], 0.6)
        self.assertNotEqual(aDCVert.index, self.v.index)
        self.assertNotEqual(aDCVert, self.v)

    def test_dcel_created_duplicate_guard(self):
        """ Check duplicate vertices aren't created, and instead vertices are reused """
        v1 = self.dc.newVertex(np.array([0.2, 0.6]))
        v2 = self.dc.newVertex(np.array([0.2, 0.6]))
        self.assertEqual(v1.index, v2.index)

    def test_dcel_created_duplicate_integers(self):
        """ Check vertex creation works with integers """
        v1 = self.dc.newVertex(np.array([3,3]))
        v2 = self.dc.newVertex(np.array([3,3]))
        self.assertEqual(v1.index, v2.index)
        
    def test_vertex_export_import(self):
        """ Check the exported vertex data has a minimum set of fields """
        exported = self.v._export()
        self.assertTrue(all([x in exported for x in ["i","x","y","halfEdges","data","active"]]))

    def test_vertex_activation(self):
        """ Check vertices can be activated and deactivated """
        self.assertTrue(self.v.active)
        self.v.deactivate()
        self.assertFalse(self.v.active)
        self.v.activate()
        self.assertTrue(self.v.active)

    def test_vertex_edge_registration_and_edgeless(self):
        """ Check vertices register halfedges correctly """
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
        """ Check vertices can create a bbox around themselves """
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
        """ Check a vertex can call through to its dcel to get nearby vertices """
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
        """ Get a vertex's coordinates as a np.ndarray """
        arr = self.v.toArray()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (2,))
        self.assertTrue(all(arr == np.array([0.2, 0.6])))
        
    def test_vertex_extend_to_line(self):
        """ Check a vertex can be extended to create a line """
        #create a line appropriately
        v1 = self.dc.newVertex(np.array([0,0]))
        e = v1.extend_line_to(target=np.array([1,0]))
        self.assertIsNotNone(e)
        self.assertIsInstance(e, dcel.HalfEdge)
        self.assertTrue(all(e.twin.origin.toArray() == np.array([1,0])))
        self.assertTrue(all(e.origin.toArray() == np.array([0,0])))
        self.assertFalse(v1.isEdgeless())
        
    def test_vertex_get_sorted_edges(self):
        """ Check edges from a vertex can be sorted ccw and returned """
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

    def test_has_constraints_false(self):
        """ A Vertex doesn't have constraints by default """
        v1 = self.dc.newVertex(np.array([0,0]))
        self.assertFalse(v1.has_constraints())

    def test_has_constraints_false_candidates(self):
        """ a vertex doesn't have a constraint with non-related candidates  """
        v1 = self.dc.newVertex(np.array([0,0]))
        self.assertFalse(v1.has_constraints(set(["a"])))

    def test_has_constraints_false(self):
        """ a vertex doesn't have a constraint with a halfedge passed in as a candidate """
        v1 = self.dc.newVertex(np.array([0,0]))
        v2 = self.dc.newVertex(np.array([0,0]))
        e = self.dc.newEdge(v1, v2)
        self.assertFalse(v1.has_constraints(set([e, e.twin])))

    def test_has_constraints_true(self):
        """ A vertex does have a constraint if the halfedge is not passed in as a candidate """
        v1 = self.dc.newVertex(np.array([0,0]))
        v2 = self.dc.newVertex(np.array([0,0]))
        e = self.dc.newEdge(v1, v2)
        self.assertTrue(v1.has_constraints())

    def test_has_constraints_two_edges_false(self):
        """ a vertex with multiple edges isnt constrained if both are passed in as candidates """  
        v1 = self.dc.newVertex(np.array([0,0]))
        v2 = self.dc.newVertex(np.array([0,0]))
        e = self.dc.newEdge(v1, v2)
        e2 = self.dc.newEdge(v1,v2)
        self.assertFalse(v1.has_constraints(set([e, e2, e.twin, e2.twin])))

    def test_has_constraints_two_edges_true(self):
        """ a vertex with multiple edges is constrained if not all its edges as candidates """  
        v1 = self.dc.newVertex(np.array([0,0]))
        v2 = self.dc.newVertex(np.array([0,0]))
        e = self.dc.newEdge(v1, v2)
        e2 = self.dc.newEdge(v1,v2)
        self.assertTrue(v1.has_constraints(set([e, e.twin])))

    def test_translate_modified(self):
        """ Check a vertex can be translated without creating a new vertex """
        v1 = self.dc.newVertex(np.array([5,5]))
        v2, edit_e = v1.translate(np.array([1,1]))
        self.assertEqual(edit_e, EditE.MODIFIED)
        self.assertEqual(v1, v2)
        self.assertTrue(np.allclose(v1.toArray(), np.array([6,6])))

    def test_translate_force_modified(self):
        """ Check translation can be forced to modify the vertex """
        e = self.dc.createEdge(np.array([5,5]), np.array([6,6]))
        v1 = e.origin
        v2, edit_e = v1.translate(np.array([1,1]), force=True)
        self.assertEqual(edit_e, EditE.MODIFIED)
        self.assertEqual(v1, v2)
        self.assertTrue(np.allclose(v1.toArray(), np.array([6,6])))

    def test_translate_new(self):
        """ check translate can create a new vertex """
        e = self.dc.createEdge(np.array([5,5]), np.array([6,6]))
        v1 = e.origin
        v2, edit_e = v1.translate(np.array([1,1]))
        self.assertEqual(edit_e, EditE.NEW)
        self.assertNotEqual(v1, v2)
        self.assertTrue(np.allclose(v2.toArray(), np.array([6,6])))
        self.assertTrue(np.allclose(v1.toArray(), np.array([5,5])))
        
    def test_translate_modified_by_candidates(self):
        """ check translate can modify if its constraints match the passed in set """
        e = self.dc.createEdge(np.array([5,5]), np.array([6,6]))
        v1 = e.origin
        v2, edit_e = v1.translate(np.array([1,1]), candidates=set([e, e.twin]))
        self.assertEqual(edit_e, EditE.MODIFIED)
        self.assertEqual(v1, v2)
        self.assertTrue(np.allclose(v2.toArray(), np.array([6,6])))

    def test_rotate_modified(self):
        """ Check a vertex can be rotated around another point """
        v1 = self.dc.newVertex(np.array([5,0]))
        v2, edit_e = v1.rotate(np.array([0,0]), radians(90))
        self.assertEqual(edit_e, EditE.MODIFIED)
        self.assertEqual(v1, v2)
        self.assertTrue(np.allclose(v2.toArray(), np.array([0,5])))
        
    def test_rotate_new(self):
        """ Check rotate can create a new vertex when constrained """
        e = self.dc.createEdge(np.array([5,0]), np.array([6,0]))
        v1 = e.twin.origin
        v2, edit_e = v1.rotate(np.array([0,0]), radians(90))
        self.assertEqual(edit_e, EditE.NEW)
        self.assertNotEqual(v1, v2)
        self.assertTrue(np.allclose(v2.toArray(), np.array([0,6])))

    def test_rotate_modified_force(self):
        """ Check rotate can be forced to modify an existing vertex """
        e = self.dc.createEdge(np.array([5,0]), np.array([6,0]))
        v1 = e.twin.origin
        v2, edit_e = v1.rotate(np.array([0,0]), radians(90), force=True)
        self.assertEqual(edit_e, EditE.MODIFIED)
        self.assertEqual(v1, v2)
        self.assertTrue(np.allclose(v2.toArray(), np.array([0,6])))

    def test_rotate_modified_due_to_candidates(self):
        """ check rotate can modify based on passed in candidate set """
        e = self.dc.createEdge(np.array([5,0]), np.array([6,0]))
        v1 = e.twin.origin
        v2, edit_e = v1.rotate(np.array([0,0]), radians(90), candidates=set([e, e.twin]))
        self.assertEqual(edit_e, EditE.MODIFIED)
        self.assertEqual(v1, v2)
        self.assertTrue(np.allclose(v2.toArray(), np.array([0,6])))
        

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
