import unittest
import logging
import IPython
import numpy as np
from random import shuffle
from math import radians
from test_context import cairo_utils as utils
from cairo_utils import dcel
from cairo_utils.math import get_distance_raw


class DCEL_ACTUAL_Tests(unittest.TestCase):
    def setUp(self):
        self.dc = dcel.DCEL()

    def tearDown(self):
        self.dc = None

    #----------
    #test dcel creation
    def test_dcel_creation(self):
        """ Check basic dcel creation """
        self.assertIsNotNone(self.dc)
        self.assertIsInstance(self.dc, dcel.DCEL)
        self.assertTrue((self.dc.bbox == np.array([-200, -200, 200, 200])).all())

    def test_dcel_creation_custom_bbox(self):
        """ Check the dcel can have a custom bbox instead of default """
        dc = dcel.DCEL(bbox=np.array([20,30, 80, 100]))
        self.assertTrue((dc.bbox == np.array([20, 30, 80, 100])).all())
        
    def test_frontier(self):
        """ Check there is a frontier to the dcel """
        self.assertEqual(len(self.dc.frontier), 0)
        self.dc.frontier.update([1,2,3,4])
        self.assertEqual(len(self.dc.frontier), 4)
        self.dc.reset_frontier()
        self.assertEqual(len(self.dc.frontier), 0)
        
    def test_copy_empty(self):
        """ Check base case copy of dcel """
        dc = self.dc.copy()
        self.assertIsInstance(dc, dcel.DCEL)
        
    def test_quad_tree(self):
        """ Check the dcel has a quad tree """
        self.assertEqual(self.dc.vertex_quad_tree.countmembers(), 0)
        self.dc.vertex_quad_tree.insert(item="blah", bbox=[1,1,2,2])
        self.assertEqual(self.dc.vertex_quad_tree.countmembers(), 1)
        self.dc.clear_quad_tree()
        self.assertEqual(self.dc.vertex_quad_tree.countmembers(), 0)

    def test_context_manager_normal(self):
        """ Check the dcel can behave as a context, creating a stack of quad trees """
        self.assertEqual(len(self.dc.quad_tree_stack), 0)
        with self.dc:
            self.assertEqual(len(self.dc.quad_tree_stack), 1)
            with self.dc:
                self.assertEqual(len(self.dc.quad_tree_stack), 2)
            self.assertEqual(len(self.dc.quad_tree_stack), 1)
        self.assertEqual(len(self.dc.quad_tree_stack), 0)

    #todo: context manager where you pass in vertices for the quad tree stack
        
    def test_context_manager_exception(self):
        """ Check exceptions don't break the context """
        with self.assertRaises(Exception):
            with self.dc:
                self.assertTrue(len(self.dc.quad_tree_stack) == 1)
                raise Exception("test")
            self.assertTrue(len(self.dc.quad_tree_stack) == 0)
        self.assertTrue(len(self.dc.quad_tree_stack) == 0)

        
    def test_vertex_creation(self):
        """ check vertices can be created through the dcel """
        self.assertEqual(len(self.dc.vertices), 0)
        self.assertEqual(self.dc.vertex_quad_tree.countmembers(), 0)
        v = self.dc.newVertex(np.array([0,0]))
        self.assertIsNotNone(v)
        self.assertIsInstance(v, dcel.Vertex)
        self.assertEqual(len(self.dc.vertices), 1)
        self.assertEqual(self.dc.vertex_quad_tree.countmembers(), 1)

    def test_halfedge_creation(self):
        """ Check the dcel can create halfedges """
        v1 = self.dc.newVertex(np.array([0,0]))
        v2 = self.dc.newVertex(np.array([1,1]))
        self.assertEqual(len(self.dc.vertices), 2)
        self.assertEqual(len(self.dc.halfEdges), 0)
        e = self.dc.newEdge(v1, v2)
        self.assertEqual(len(self.dc.halfEdges), 2)
        self.assertIsNotNone(e)
        self.assertIsInstance(e, dcel.HalfEdge)

    def test_halfedge_creation_utility(self):
        """ Check the dcel can create halfedges from raw coordinates """
        self.assertEqual(len(self.dc.vertices), 0)
        self.assertEqual(len(self.dc.halfEdges), 0)
        e = self.dc.createEdge(np.array([0,0]), np.array([1,1]))
        self.assertEqual(len(self.dc.vertices), 2)
        self.assertEqual(len(self.dc.halfEdges), 2)

    def test_face_creation(self):
        """ Check the dcel can create faces """
        self.assertEqual(len(self.dc.faces), 0)
        f = self.dc.newFace()
        self.assertEqual(len(self.dc.faces), 1)
        self.assertIsNotNone(f)
        self.assertIsInstance(f, dcel.Face)

    def test_path_creation(self):
        """ Check the dcel can create a sequence of edges """
        coords = np.array([[0,0],[1,0],[0,1]])
        self.assertEqual(len(self.dc.vertices), 0)
        self.assertEqual(len(self.dc.halfEdges), 0)
        path = self.dc.createPath(coords)
        self.assertEqual(len(self.dc.vertices), 3)
        self.assertEqual(len(self.dc.halfEdges), 4)
        self.assertIsInstance(path, list)

    def test_path_cycle_creation(self):
        """ Check the dcel can create a cycle of edges """
        coords = np.array([[0,0],[1,0],[0,1]])
        self.assertEqual(len(self.dc.vertices), 0)
        self.assertEqual(len(self.dc.halfEdges), 0)
        path = self.dc.createPath(coords, close=True)
        self.assertEqual(len(self.dc.vertices), 3)
        self.assertEqual(len(self.dc.halfEdges), 6)
        self.assertIsInstance(path, list)
        

    def test_edge_linking(self):
        """ Check that the dcel can link a set of edges into a sequence """
        #create a number of edges, that share vertices
        e1 = self.dc.createEdge(np.array([0,0]), np.array([1,1]))
        e2 = self.dc.createEdge(np.array([1,1]), np.array([2,2]))
        e3 = self.dc.createEdge(np.array([2,2]), np.array([0,0]))
        #link them together
        self.dc.linkEdgesTogether([e1, e2, e3])
        #verify:
        self.assertTrue(e1.next == e2)
        self.assertTrue(e2.next == e3)
        self.assertTrue(e3.next == None)

    def test_edge_linking_loop(self):
        """ Check that the dcel can link a set of edges into a loop """
        #create a number of edges, that share vertices
        e1 = self.dc.createEdge(np.array([0,0]), np.array([1,1]))
        e2 = self.dc.createEdge(np.array([1,1]), np.array([2,2]))
        e3 = self.dc.createEdge(np.array([2,2]), np.array([0,0]))
        #link them together
        self.dc.linkEdgesTogether([e1, e2, e3], loop=True)
        #verify:
        self.assertTrue(e1.next == e2)
        self.assertTrue(e2.next == e3)
        self.assertTrue(e3.next == e1)
        
    def test_edge_linking_fail(self):
        """ Check the linking of edges can fail on bad links """
        #create a number of edges, that share vertices
        e1 = self.dc.createEdge(np.array([0,0]), np.array([1,1]))
        e2 = self.dc.createEdge(np.array([1,1]), np.array([2,2]))
        e3 = self.dc.createEdge(np.array([2,2]), np.array([0,0]))
        e2.next = 2
        #link them together
        with self.assertRaises(Exception):
            self.dc.linkEdgesTogether([e1, e2, e3], loop=True)

    def test_vertex_ordering(self):
        #create a number of vertices
        increment = int(300 / 20)
        verts = []
        for x in range(0,300, increment):
            rads = radians(x)
            coords = np.array([np.cos(rads), np.sin(rads)])
            verts.append(self.dc.newVertex(coords))
        verts_shuffled = verts.copy()
        shuffle(verts_shuffled)
        #order the shuffled verts:
        reordered_verts = self.dc.orderVertices(np.array([0,0]), verts_shuffled)
        for a,b in zip(verts, reordered_verts):
            self.assertTrue(a==b)

        
    def test_circle_constrain_vertices(self):
        v1 = self.dc.newVertex(np.array([0,0]))
        v2 = self.dc.newVertex(np.array([2,0]))
        v3 = self.dc.newVertex(np.array([3,3]))
        v4 = self.dc.newVertex(np.array([-2,-2]))
        self.dc.constrain_to_circle(np.array([0,0]), 1.0)
        self.assertFalse(v1.markedForCleanup)
        self.assertTrue(all([x.markedForCleanup for x in [v2, v3, v4]]))
        

    def test_circle_constrain_faces(self):
        central_loc = np.array([10,0])
        f = self.dc.newFace(coords=np.array([[10,0],[12,0],[10,2]]))
        original_verts = f.get_all_vertices()
        asArray = np.array([x.toArray() for x in original_verts])
        self.assertFalse((get_distance_raw(asArray, central_loc) <= 2).all())
        self.dc.constrain_to_circle(central_loc, 1.0)
        self.assertEqual(len(original_verts.symmetric_difference(f.get_all_vertices())), 0)
        self.assertEqual(len(f.edgeList), 2)
        new_verts = f.get_all_vertices()
        new_asArray = np.array([x.toArray() for x in new_verts])
        self.assertTrue((get_distance_raw(new_asArray, central_loc) <= 2).all())

    def test_circle_constrain(self):
        e1 = self.dc.createEdge(np.array([0,0]), np.array([1,0]))
        originalVertex_indices = set([x.index for x in e1.getVertices()])
        #create edges, constrain to within a radius
        self.assertTrue(np.allclose(e1.twin.origin.toArray(), np.array([1,0])))
        self.dc.constrain_to_circle(np.array([0,0]), 0.5)
        self.assertFalse(np.allclose(e1.twin.origin.toArray(), np.array([1,0])))
        self.assertTrue(np.allclose(e1.twin.origin.toArray(), np.array([0.5,0])))
        self.assertTrue(originalVertex_indices == set([x.index for x in e1.getVertices()]))
        
    def test_circle_constrain_no_op(self):
        e1 = self.dc.createEdge(np.array([0,0]), np.array([0.3,0]))
        originalVertex_indices = [x.index for x in e1.getVertices()]
        #create edges, constrain to within a radius
        self.assertTrue(np.allclose(e1.twin.origin.toArray(), np.array([0.3,0])))
        self.dc.constrain_to_circle(np.array([0,0]), 0.5)
        self.assertTrue(np.allclose(e1.twin.origin.toArray(), np.array([0.3,0])))
        self.assertTrue(originalVertex_indices == [x.index for x in e1.getVertices()])
        
    def test_circle_constrain_mark_out_of_bounds(self):
        e1 = self.dc.createEdge(np.array([2,2]), np.array([2,3]))
        #create edges, constrain to within a radius
        self.assertFalse(e1.markedForCleanup)
        self.dc.constrain_to_circle(np.array([0,0]), 0.5)
        self.assertTrue(e1.markedForCleanup)

    def test_purge_edges(self):
        e1 = self.dc.createEdge(np.array([0,0]), np.array([1,1]))
        e1Verts = e1.getVertices()
        e2 = self.dc.createEdge(np.array([2,2]), np.array([3,3]))
        e1.markForCleanup()
        self.assertTrue(e1 in self.dc.halfEdges)
        self.assertTrue(e1.twin in self.dc.halfEdges)
        self.assertTrue(e2 in self.dc.halfEdges)
        self.assertTrue(e2.twin in self.dc.halfEdges)
        self.assertTrue(e1 in e1Verts[0].halfEdges)
        self.dc.purge()
        #Only the halfedge is purged, not its twin
        self.assertFalse(e1 in self.dc.halfEdges)
        self.assertFalse(e1.twin in self.dc.halfEdges)
        self.assertTrue(e2 in self.dc.halfEdges)
        self.assertTrue(e2.twin in self.dc.halfEdges)
        self.assertFalse(e1 in e1Verts[0].halfEdges)

    def test_purge_infinite_edges(self):
        e1 = self.dc.createEdge(np.array([0,0]), np.array([1,1]))
        e1Verts = e1.getVertices()
        e2 = self.dc.createEdge(np.array([2,2]), np.array([3,3]))
        f = self.dc.newFace()
        f.add_edges([e1, e2])
        e1.markForCleanup()
        self.assertTrue(e1 in self.dc.halfEdges)
        self.assertTrue(e1.twin in self.dc.halfEdges)
        self.assertTrue(e2 in self.dc.halfEdges)
        self.assertTrue(e2.twin in self.dc.halfEdges)
        self.assertTrue(e1 in e1Verts[0].halfEdges)
        self.assertTrue(e1.face == f)
        self.assertTrue(e2.face == f)
        self.dc.purge()
        #Only the halfedge is purged, not its twin
        self.assertFalse(e1 in self.dc.halfEdges)
        self.assertFalse(e1.twin in self.dc.halfEdges)
        self.assertTrue(e2 in self.dc.halfEdges)
        self.assertTrue(e2.twin in self.dc.halfEdges)
        self.assertFalse(e1 in e1Verts[0].halfEdges)
        self.assertFalse(e1.face == f)
        self.assertTrue(e2.face == f)
        self.assertFalse(e1 in f.edgeList)
        self.assertTrue(e2 in f.edgeList)

    def test_purge_vertices(self):
        v1 = self.dc.newVertex(np.array([0,0]))
        v2 = self.dc.newVertex(np.array([1,1]))
        v3 = self.dc.newVertex(np.array([0,1]))
        e = self.dc.newEdge(v1,v2)
        e_twin = e.twin
        v1.markForCleanup()
        self.dc.purge()
        self.assertFalse(v1 in self.dc.vertices)
        self.assertFalse(v2 in self.dc.vertices)
        self.assertFalse(e in self.dc.halfEdges)
        self.assertFalse(e_twin in self.dc.halfEdges)
        self.assertTrue(v3 in self.dc.vertices)
        

    def test_purge_faces(self):
        f = self.dc.newFace(coords=np.array([[0,0],[0,1],[1,0]]))
        f.markForCleanup()
        edges = self.dc.halfEdges.copy()
        self.assertEqual(len(edges), 6)
        verts = self.dc.vertices.copy()
        self.assertEqual(len(verts), 3)
        self.dc.purge()
        self.assertEqual(len(self.dc.vertices), 0)
        self.assertEqual(len(self.dc.halfEdges), 0)
        self.assertEqual(len(self.dc.faces), 0)

    def test_purge_nothing(self):
        f = self.dc.newFace(coords=np.array([[0,0],[0,1],[1,0]]))
        self.dc.purge()
        self.assertEqual(len(self.dc.vertices), 3)
        self.assertEqual(len(self.dc.halfEdges), 6)
        self.assertEqual(len(self.dc.faces), 1)
                                
        
    def test_export_vertices(self):
        self.dc.newVertex(np.array([0,0]))
        self.dc.newVertex(np.array([1,1]))
        exportedData = self.dc.export_data()
        self.assertTrue('vertices' in exportedData)
        self.assertEqual(len(exportedData['vertices']), 2)
        for x in exportedData['vertices']:
            self.assertTrue(all([y in x for y in ['i','x','y','halfEdges','data','active']]))
        
    def test_export_halfedges(self):
        self.dc.createEdge(np.array([0,0]), np.array([1,1]))
        self.dc.createEdge(np.array([2,2]), np.array([3,3]))
        exportedData = self.dc.export_data()
        self.assertTrue('vertices' in exportedData)
        self.assertEqual(len(exportedData['vertices']), 4)
        self.assertTrue('halfEdges' in exportedData)
        self.assertEqual(len(exportedData['halfEdges']), 4)
        for x in exportedData['halfEdges']:
            self.assertTrue(all([y in x for y in ['i','origin','twin','face','next','prev','data']]))
        

    def test_export_faces(self):
        self.dc.newFace(np.array([0,0]))
        self.dc.newFace(np.array([1,1]))
        exportedData = self.dc.export_data()
        self.assertTrue('faces' in exportedData)
        self.assertEqual(len(exportedData['faces']), 2)
        for x in exportedData['faces']:
            self.assertTrue(all([y in x for y in ['i','edges','sitex','sitey','data']]))
        

    def test_import_vertices(self):
        testData = {
            'vertices' : [{'i': 5, 'x': 3, 'y':4, 'halfEdges': [], 'data':{}, 'active': True },
            {'i': 10,'x':10,'y':5, 'halfEdges':[], 'data':{}, 'active':False }],
            'halfEdges' : [],
            'faces' : [],
            'bbox' : np.array([0,0,10,10])           
        }
        self.dc.import_data(testData)
        self.assertEqual(len(self.dc.vertices), 2)
        self.assertEqual(set([5,10]), set([x.index for x in self.dc.vertices]))
        

    def test_import_halfEdges(self):
        testData = {
            'vertices' : [{'i': 5, 'x': 3, 'y':4, 'halfEdges': [], 'data':{}, 'active': True },
            {'i': 10,'x':10,'y':5, 'halfEdges':[], 'data':{}, 'active':False }],
            'halfEdges' : [],
            'faces' : [],
            'bbox' : np.array([0,0,10,10])           
        }
        self.dc.import_data(testData)
        self.assertEqual(len(self.dc.vertices), 2)
        self.assertEqual(set([5,10]), set([x.index for x in self.dc.vertices]))

    def test_import_faces(self):
        testData = {
            'vertices' : [{'i': 5, 'x': 3, 'y':4, 'halfEdges': [], 'data':{}, 'active': True },
            {'i': 10,'x':10,'y':5, 'halfEdges':[], 'data':{}, 'active':False }],
            'halfEdges' : [],
            'faces' : [],
            'bbox' : np.array([0,0,10,10])           
        }
        self.dc.import_data(testData)
        self.assertEqual(len(self.dc.vertices), 2)
        self.assertEqual(set([5,10]), set([x.index for x in self.dc.vertices]))

    
    def test_save_load(self):
        self.dc.newFace()
        self.dc.newFace(np.array([1,1]))
        self.dc.createEdge(np.array([0,0]), np.array([1,1]))
        self.dc.createEdge(np.array([2,2]), np.array([3,3]))
        self.dc.savefile("dcel_actual_save_test")

        newDCEL = dcel.DCEL.loadfile("dcel_actual_save_test")
        self.assertEqual(len(newDCEL.vertices), 4)
        self.assertEqual(len(newDCEL.halfEdges), 4)
        self.assertEqual(len(newDCEL.faces), 2)        

    def test_force_edge_lengths(self):
        e = self.dc.createEdge(np.array([0,0]), np.array([10,0]))
        self.assertEqual(e.getLength_sq(), (pow(10,2)))
        self.dc.force_all_edge_lengths(pow(2,2))
        self.assertTrue(all([x.getLength_sq() <= (pow(2,2)) for x in self.dc.halfEdges]))
        self.assertEqual(len(self.dc.halfEdges), 8 * 2)

    def test_intersect_halfedges(self):
        self.assertTrue(False)
        
        
if __name__ == "__main__":
      #use python $filename to use this logging setup
      LOGLEVEL = logging.INFO
      logFileName = "log.DCEL_ACTUAL_Tests"
      logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
      console = logging.StreamHandler()
      console.setLevel(logging.WARN)
      logging.getLogger().addHandler(console)
      unittest.main()
      #reminder: user logging.getLogger().setLevel(logging.NOTSET) for log control
