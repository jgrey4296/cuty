import unittest
import logging

import numpy as np
from random import shuffle
from math import radians
from test_context import cairo_utils as utils
from cairo_utils import dcel
from cairo_utils.umath import get_distance_raw


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
        v = self.dc.new_vertex(np.array([0,0]))
        self.assertIsNotNone(v)
        self.assertIsInstance(v, dcel.Vertex)
        self.assertEqual(len(self.dc.vertices), 1)
        self.assertEqual(self.dc.vertex_quad_tree.countmembers(), 1)

    def test_halfedge_creation(self):
        """ Check the dcel can create half_edges """
        v1 = self.dc.new_vertex(np.array([0,0]))
        v2 = self.dc.new_vertex(np.array([1,1]))
        self.assertEqual(len(self.dc.vertices), 2)
        self.assertEqual(len(self.dc.half_edges), 0)
        e = self.dc.new_edge(v1, v2)
        self.assertEqual(len(self.dc.half_edges), 2)
        self.assertIsNotNone(e)
        self.assertIsInstance(e, dcel.HalfEdge)

    def test_halfedge_creation_utility(self):
        """ Check the dcel can create half_edges from raw coordinates """
        self.assertEqual(len(self.dc.vertices), 0)
        self.assertEqual(len(self.dc.half_edges), 0)
        e = self.dc.create_edge(np.array([0,0]), np.array([1,1]))
        self.assertEqual(len(self.dc.vertices), 2)
        self.assertEqual(len(self.dc.half_edges), 2)

    def test_face_creation(self):
        """ Check the dcel can create faces """
        self.assertEqual(len(self.dc.faces), 0)
        f = self.dc.new_face()
        self.assertEqual(len(self.dc.faces), 1)
        self.assertIsNotNone(f)
        self.assertIsInstance(f, dcel.Face)

    def test_path_creation(self):
        """ Check the dcel can create a sequence of edges """
        coords = np.array([[0,0],[1,0],[0,1]])
        self.assertEqual(len(self.dc.vertices), 0)
        self.assertEqual(len(self.dc.half_edges), 0)
        path = self.dc.create_path(coords)
        self.assertEqual(len(self.dc.vertices), 3)
        self.assertEqual(len(self.dc.half_edges), 4)
        self.assertIsInstance(path, list)

    def test_path_cycle_creation(self):
        """ Check the dcel can create a cycle of edges """
        coords = np.array([[0,0],[1,0],[0,1]])
        self.assertEqual(len(self.dc.vertices), 0)
        self.assertEqual(len(self.dc.half_edges), 0)
        path = self.dc.create_path(coords, close=True)
        self.assertEqual(len(self.dc.vertices), 3)
        self.assertEqual(len(self.dc.half_edges), 6)
        self.assertIsInstance(path, list)

    def test_edge_linking(self):
        """ Check that the dcel can link a set of edges into a sequence """
        #create a number of edges, that share vertices
        e1 = self.dc.create_edge(np.array([0,0]), np.array([1,1]))
        e2 = self.dc.create_edge(np.array([1,1]), np.array([2,2]))
        e3 = self.dc.create_edge(np.array([2,2]), np.array([0,0]))
        #link them together
        self.dc.link_edges_together([e1, e2, e3])
        #verify:
        self.assertTrue(e1.next == e2)
        self.assertTrue(e2.next == e3)
        self.assertTrue(e3.next == None)

    def test_edge_linking_loop(self):
        """ Check that the dcel can link a set of edges into a loop """
        #create a number of edges, that share vertices
        e1 = self.dc.create_edge(np.array([0,0]), np.array([1,1]))
        e2 = self.dc.create_edge(np.array([1,1]), np.array([2,2]))
        e3 = self.dc.create_edge(np.array([2,2]), np.array([0,0]))
        #link them together
        self.dc.link_edges_together([e1, e2, e3], loop=True)
        #verify:
        self.assertTrue(e1.next == e2)
        self.assertTrue(e2.next == e3)
        self.assertTrue(e3.next == e1)

    def test_edge_linking_fail(self):
        """ Check the linking of edges can fail on bad links """
        #create a number of edges, that share vertices
        e1 = self.dc.create_edge(np.array([0,0]), np.array([1,1]))
        e2 = self.dc.create_edge(np.array([1,1]), np.array([2,2]))
        e3 = self.dc.create_edge(np.array([2,2]), np.array([0,0]))
        e2.next = 2
        #link them together
        with self.assertRaises(Exception):
            self.dc.link_edges_together([e1, e2, e3], loop=True)

    def test_vertex_ordering(self):
        #create a number of vertices
        increment = int(300 / 20)
        verts = []
        for x in range(0,300, increment):
            rads = radians(x)
            coords = np.array([np.cos(rads), np.sin(rads)])
            verts.append(self.dc.new_vertex(coords))
        verts_shuffled = verts.copy()
        shuffle(verts_shuffled)
        #order the shuffled verts:
        reordered_verts = self.dc.order_vertices(np.array([0,0]), verts_shuffled)
        for a,b in zip(verts, reordered_verts):
            self.assertTrue(a==b)

    def test_circle_constrain_vertices(self):
        v1 = self.dc.new_vertex(np.array([0,0]))
        v2 = self.dc.new_vertex(np.array([2,0]))
        v3 = self.dc.new_vertex(np.array([3,3]))
        v4 = self.dc.new_vertex(np.array([-2,-2]))
        self.dc.constrain_to_circle(np.array([0,0]), 1.0)
        self.assertFalse(v1.marked_for_cleanup)
        self.assertTrue(all([x.marked_for_cleanup for x in [v2, v3, v4]]))

    def test_circle_constrain_faces(self):
        central_loc = np.array([10,0])
        f = self.dc.new_face(coords=np.array([[10,0],[12,0],[10,2]]))
        original_verts = f.get_all_vertices()
        asArray = np.array([x.to_array() for x in original_verts])
        self.assertFalse((get_distance_raw(asArray, central_loc) <= 2).all())
        self.dc.constrain_to_circle(central_loc, 1.0)
        self.assertEqual(len(original_verts.symmetric_difference(f.get_all_vertices())), 0)
        self.assertEqual(len(f.edge_list), 2)
        new_verts = f.get_all_vertices()
        new_asArray = np.array([x.to_array() for x in new_verts])
        self.assertTrue((get_distance_raw(new_asArray, central_loc) <= 2).all())

    def test_circle_constrain(self):
        e1 = self.dc.create_edge(np.array([0,0]), np.array([1,0]))
        originalVertex_indices = set([x.index for x in e1.get_vertices()])
        #create edges, constrain to within a radius
        self.assertTrue(np.allclose(e1.twin.origin.to_array(), np.array([1,0])))
        self.dc.constrain_to_circle(np.array([0,0]), 0.5)
        self.assertFalse(np.allclose(e1.twin.origin.to_array(), np.array([1,0])))
        self.assertTrue(np.allclose(e1.twin.origin.to_array(), np.array([0.5,0])))
        self.assertTrue(originalVertex_indices == set([x.index for x in e1.get_vertices()]))

    def test_circle_constrain_no_op(self):
        e1 = self.dc.create_edge(np.array([0,0]), np.array([0.3,0]))
        originalVertex_indices = [x.index for x in e1.get_vertices()]
        #create edges, constrain to within a radius
        self.assertTrue(np.allclose(e1.twin.origin.to_array(), np.array([0.3,0])))
        self.dc.constrain_to_circle(np.array([0,0]), 0.5)
        self.assertTrue(np.allclose(e1.twin.origin.to_array(), np.array([0.3,0])))
        self.assertTrue(originalVertex_indices == [x.index for x in e1.get_vertices()])

    def test_circle_constrain_mark_out_of_bounds(self):
        e1 = self.dc.create_edge(np.array([2,2]), np.array([2,3]))
        #create edges, constrain to within a radius
        self.assertFalse(e1.marked_for_cleanup)
        self.dc.constrain_to_circle(np.array([0,0]), 0.5)
        self.assertTrue(e1.marked_for_cleanup)

    def test_purge_edges(self):
        e1 = self.dc.create_edge(np.array([0,0]), np.array([1,1]))
        e1Verts = e1.get_vertices()
        e2 = self.dc.create_edge(np.array([2,2]), np.array([3,3]))
        e1.mark_for_cleanup()
        self.assertTrue(e1 in self.dc.half_edges)
        self.assertTrue(e1.twin in self.dc.half_edges)
        self.assertTrue(e2 in self.dc.half_edges)
        self.assertTrue(e2.twin in self.dc.half_edges)
        self.assertTrue(e1 in e1Verts[0].half_edges)
        self.dc.purge()
        #Only the halfedge is purged, not its twin
        self.assertFalse(e1 in self.dc.half_edges)
        self.assertFalse(e1.twin in self.dc.half_edges)
        self.assertTrue(e2 in self.dc.half_edges)
        self.assertTrue(e2.twin in self.dc.half_edges)
        self.assertFalse(e1 in e1Verts[0].half_edges)

    def test_purge_infinite_edges(self):
        e1 = self.dc.create_edge(np.array([0,0]), np.array([1,1]))
        e1Verts = e1.get_vertices()
        e2 = self.dc.create_edge(np.array([2,2]), np.array([3,3]))
        f = self.dc.new_face()
        f.add_edges([e1, e2])
        e1.mark_for_cleanup()
        self.assertTrue(e1 in self.dc.half_edges)
        self.assertTrue(e1.twin in self.dc.half_edges)
        self.assertTrue(e2 in self.dc.half_edges)
        self.assertTrue(e2.twin in self.dc.half_edges)
        self.assertTrue(e1 in e1Verts[0].half_edges)
        self.assertTrue(e1.face == f)
        self.assertTrue(e2.face == f)
        self.dc.purge()
        #Only the halfedge is purged, not its twin
        self.assertFalse(e1 in self.dc.half_edges)
        self.assertFalse(e1.twin in self.dc.half_edges)
        self.assertTrue(e2 in self.dc.half_edges)
        self.assertTrue(e2.twin in self.dc.half_edges)
        self.assertFalse(e1 in e1Verts[0].half_edges)
        self.assertFalse(e1.face == f)
        self.assertTrue(e2.face == f)
        self.assertFalse(e1 in f.edge_list)
        self.assertTrue(e2 in f.edge_list)

    def test_purge_vertices(self):
        v1 = self.dc.new_vertex(np.array([0,0]))
        v2 = self.dc.new_vertex(np.array([1,1]))
        v3 = self.dc.new_vertex(np.array([0,1]))
        e = self.dc.new_edge(v1,v2)
        e_twin = e.twin
        v1.mark_for_cleanup()
        self.dc.purge()
        self.assertFalse(v1 in self.dc.vertices)
        self.assertFalse(v2 in self.dc.vertices)
        self.assertFalse(e in self.dc.half_edges)
        self.assertFalse(e_twin in self.dc.half_edges)
        self.assertTrue(v3 in self.dc.vertices)

    def test_purge_faces(self):
        f = self.dc.new_face(coords=np.array([[0,0],[0,1],[1,0]]))
        f.mark_for_cleanup()
        edges = self.dc.half_edges.copy()
        self.assertEqual(len(edges), 6)
        verts = self.dc.vertices.copy()
        self.assertEqual(len(verts), 3)
        self.dc.purge()
        self.assertEqual(len(self.dc.vertices), 0)
        self.assertEqual(len(self.dc.half_edges), 0)
        self.assertEqual(len(self.dc.faces), 0)

    def test_purge_nothing(self):
        f = self.dc.new_face(coords=np.array([[0,0],[0,1],[1,0]]))
        self.dc.purge()
        self.assertEqual(len(self.dc.vertices), 3)
        self.assertEqual(len(self.dc.half_edges), 6)
        self.assertEqual(len(self.dc.faces), 1)

    def test_export_half_edges(self):
        self.dc.create_edge(np.array([0,0]), np.array([1,1]))
        self.dc.create_edge(np.array([2,2]), np.array([3,3]))
        exportedData = self.dc.export_data()
        self.assertTrue('vertices' in exportedData)
        self.assertEqual(len(exportedData['vertices']), 4)
        self.assertTrue('half_edges' in exportedData)
        self.assertEqual(len(exportedData['half_edges']), 4)
        for x in exportedData['half_edges']:
            self.assertTrue(all([y in x for y in ['i','origin','twin','face','next','prev','enum_data', 'non_enum_data']]))

    def test_export_faces(self):
        self.dc.new_face(np.array([0,0]))
        self.dc.new_face(np.array([1,1]))
        exportedData = self.dc.export_data()
        self.assertTrue('faces' in exportedData)
        self.assertEqual(len(exportedData['faces']), 2)
        for x in exportedData['faces']:
            self.assertTrue(all([y in x for y in ['i','edges','sitex','sitey','enum_data', 'non_enum_data']]))

    def test_import_vertices(self):
        testData = {
            'vertices' : [{'i': 5, 'x': 3, 'y':4, 'half_edges': [], 'enum_data':{},
                           'non_enum_data': {}, 'active': True },
            {'i': 10,'x':10,'y':5, 'half_edges':[], 'enum_data':{},
             'non_enum_data': {}, 'active':False }],
            'half_edges' : [],
            'faces' : [],
            'bbox' : np.array([0,0,10,10])
        }
        self.dc.import_data(testData)
        self.assertEqual(len(self.dc.vertices), 2)
        self.assertEqual(set([5,10]), set([x.index for x in self.dc.vertices]))

    def test_import_half_edges(self):
        testData = {
            'vertices' : [{'i': 5, 'x': 3, 'y':4, 'half_edges': [], 'enum_data':{},
                           'non_enum_data': {}, 'active': True },
            {'i': 10,'x':10,'y':5, 'half_edges':[], 'enum_data':{},
                           'non_enum_data': {}, 'active':False }],
            'half_edges' : [],
            'faces' : [],
            'bbox' : np.array([0,0,10,10])
        }
        self.dc.import_data(testData)
        self.assertEqual(len(self.dc.vertices), 2)
        self.assertEqual(set([5,10]), set([x.index for x in self.dc.vertices]))

    def test_import_faces(self):
        testData = {
            'vertices' : [{'i': 5, 'x': 3, 'y':4, 'half_edges': [], 'enum_data':{},
                           'non_enum_data': {}, 'active': True },
                          {'i': 10,'x':10,'y':5, 'half_edges':[], 'enum_data':{},
                           'non_enum_data': {}, 'active':False }],
            'half_edges' : [],
            'faces' : [],
            'bbox' : np.array([0,0,10,10])
        }
        self.dc.import_data(testData)
        self.assertEqual(len(self.dc.vertices), 2)
        self.assertEqual(set([5,10]), set([x.index for x in self.dc.vertices]))

    def test_save_load(self):
        self.dc.new_face()
        self.dc.new_face(np.array([1,1]))
        self.dc.create_edge(np.array([0,0]), np.array([1,1]))
        self.dc.create_edge(np.array([2,2]), np.array([3,3]))
        self.dc.savefile("dcel_actual_save_test")

        newDCEL = dcel.DCEL.loadfile("dcel_actual_save_test")
        self.assertEqual(len(newDCEL.vertices), 4)
        self.assertEqual(len(newDCEL.half_edges), 4)
        self.assertEqual(len(newDCEL.faces), 2)

    def test_force_edge_lengths(self):
        e = self.dc.create_edge(np.array([0,0]), np.array([10,0]))
        self.assertEqual(e.get_length_sq(), (pow(10,2)))
        self.dc.force_all_edge_lengths(pow(2,2))
        self.assertTrue(all([x.get_length_sq() <= (pow(2,2)) for x in self.dc.half_edges]))
        self.assertEqual(len(self.dc.half_edges), 8 * 2)

    def test_intersect_half_edges_simple(self):
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

    def test_intersect_half_edges_non_flat(self):
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

    def test_intersect_half_edges_no_intersections(self):
        #create
        e1 = self.dc.create_edge(np.array([0,0]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([0,1]), np.array([1,2]))
        e3 = self.dc.create_edge(np.array([-2,0]), np.array([0,4]))
        #intersect
        results = self.dc.intersect_half_edges()
        #verify
        self.assertEqual(len(results), 0)

    def test_intersect_half_edges_endpoints(self):
        e1 = self.dc.create_edge(np.array([0,0]), np.array([1,0]))
        e2 = self.dc.create_edge(np.array([1,0]), np.array([1,1]))
        e3 = self.dc.create_edge(np.array([1,1]), np.array([0,0]))
        results = self.dc.intersect_half_edges()
        self.assertEqual(len(results), 3)

    def test_intersect_three_vert_one_horizontal(self):
        v1 = self.dc.create_edge(np.array([0,0]), np.array([0,2000]))
        v2 = self.dc.create_edge(np.array([1000,0]), np.array([1000,2000]))
        v3 = self.dc.create_edge(np.array([2000,0]), np.array([2000,2000]))

        h1 = self.dc.create_edge(np.array([0,2000]), np.array([2000,2000]))
        results = self.dc.intersect_half_edges(edge_set=set([v1,v2,v3,h1]))
        self.assertEqual(len(results), 3)



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
