import unittest
import logging
import IPython
import numpy as np
from test_context import cairo_utils as utils
from cairo_utils import dcel


class DCEL_ACTUAL_Tests(unittest.TestCase):
    def setUp(self):
        self.dc = dcel.DCEL()

    def tearDown(self):
        self.dc = None

    #----------
    #test dcel creation
    def test_dcel_creation(self):
        self.assertIsNotNone(self.dc)
        self.assertIsInstance(self.dc, dcel.DCEL)
        self.assertTrue((self.dc.bbox == np.array([-200, -200, 200, 200])).all())

    def test_dcel_creation_custom_bbox(self):
        dc = dcel.DCEL(bbox=np.array([20,30, 80, 100]))
        self.assertTrue((dc.bbox == np.array([20, 30, 80, 100])).all())
        
    def test_frontier(self):
        self.assertEqual(len(self.dc.frontier), 0)
        self.dc.frontier.update([1,2,3,4])
        self.assertEqual(len(self.dc.frontier), 4)
        self.dc.reset_frontier()
        self.assertEqual(len(self.dc.frontier), 0)

    def test_copy_empty(self):
        dc = self.dc.copy()
        self.assertIsInstance(dc, dcel.DCEL)

    def test_quad_tree(self):
        self.assertEqual(self.dc.vertex_quad_tree.countmembers(), 0)
        self.dc.vertex_quad_tree.insert(item="blah", bbox=[1,1,2,2])
        self.assertEqual(self.dc.vertex_quad_tree.countmembers(), 1)
        self.dc.clear_quad_tree()
        self.assertEqual(self.dc.vertex_quad_tree.countmembers(), 0)
    
    def test_vertex_creation(self):
        self.assertEqual(len(self.dc.vertices), 0)
        self.assertEqual(self.dc.vertex_quad_tree.countmembers(), 0)
        v = self.dc.newVertex(np.array([0,0]))
        self.assertIsNotNone(v)
        self.assertIsInstance(v, dcel.Vertex)
        self.assertEqual(len(self.dc.vertices), 1)
        self.assertEqual(self.dc.vertex_quad_tree.countmembers(), 1)

    def test_halfedge_creation(self):
        v1 = self.dc.newVertex(np.array([0,0]))
        v2 = self.dc.newVertex(np.array([1,1]))
        self.assertEqual(len(self.dc.vertices), 2)
        self.assertEqual(len(self.dc.halfEdges), 0)
        e = self.dc.newEdge(v1, v2)
        self.assertEqual(len(self.dc.halfEdges), 2)
        self.assertIsNotNone(e)
        self.assertIsInstance(e, dcel.HalfEdge)

    def test_halfedge_creation_utility(self):
        self.assertEqual(len(self.dc.vertices), 0)
        self.assertEqual(len(self.dc.halfEdges), 0)
        e = self.dc.createEdge(np.array([0,0]), np.array([1,1]))
        self.assertEqual(len(self.dc.vertices), 2)
        self.assertEqual(len(self.dc.halfEdges), 2)

    def test_face_creation(self):
        self.assertEqual(len(self.dc.faces), 0)
        f = self.dc.newFace()
        self.assertEqual(len(self.dc.faces), 1)
        self.assertIsNotNone(f)
        self.assertIsInstance(f, dcel.Face)

    def test_path_creation(self):
        coords = np.array([[0,0],[1,0],[0,1]])
        self.assertEqual(len(self.dc.vertices), 0)
        self.assertEqual(len(self.dc.halfEdges), 0)
        path = self.dc.createPath(coords)
        self.assertEqual(len(self.dc.vertices), 3)
        self.assertEqual(len(self.dc.halfEdges), 4)
        self.assertIsInstance(path, list)

    def test_path_cycle_creation(self):
        coords = np.array([[0,0],[1,0],[0,1]])
        self.assertEqual(len(self.dc.vertices), 0)
        self.assertEqual(len(self.dc.halfEdges), 0)
        path = self.dc.createPath(coords, close=True)
        self.assertEqual(len(self.dc.vertices), 3)
        self.assertEqual(len(self.dc.halfEdges), 6)
        self.assertIsInstance(path, list)
        

    def test_edge_linking(self):
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
        #create a number of edges, that share vertices
        e1 = self.dc.createEdge(np.array([0,0]), np.array([1,1]))
        e2 = self.dc.createEdge(np.array([1,1]), np.array([2,2]))
        e3 = self.dc.createEdge(np.array([2,2]), np.array([0,0]))
        e2.next = 2
        #link them together
        with self.assertRaises(Exception):
            self.dc.linkEdgesTogether([e1, e2, e3], loop=True)

    def test_face_edge_loop(self):
        #create edges
        #create a face
        #combine
        return 0

    def test_vertex_ordering(self):
        #create a number of vertices
        #order them
        return 0
        
    def test_constraints(self):
        #create edges, that go outside the bounding box
        #run
        #verify they are constrained
        return 0
        
    def test_circle_constrain(self):
        #create edges, constrain to within a radius
        return 0
        
    def test_purging(self):
        #create vertices, halfedges, faces
        #mark to purge
        #check they are deleted
        return 0
        
    def test_complete_faces(self):
        #create a face with an unconnected edge
        #complete it
        #verify
        return 0
        
    def test_edge_connections(self):
        #add corners to edges
        return 0
        
    def test_corner_vertices(self):
        #create vertices at the corners of the bbox
        return 0
        
    def test_halfedge_fixup(self):
        return 0

    def test_verification(self):
        #verify vertexs, halfedges, faces appropriately
        return 0

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
