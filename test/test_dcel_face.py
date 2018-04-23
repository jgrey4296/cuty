import unittest
import logging
import numpy as np
import IPython
from random import shuffle
from math import radians
from itertools import islice, cycle
from test_context import cairo_utils as utils
from cairo_utils import dcel
from cairo_utils.math import get_distance_raw
from cairo_utils.dcel.constants import EditE

class DCEL_FACE_Tests(unittest.TestCase):
    def setUp(self):
        self.dc = dcel.DCEL()
        self.verts = [self.dc.newVertex(x) for x in np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])]
        self.hes = [self.dc.newEdge(a,b) for a,b in zip(self.verts, islice(cycle(self.verts), 1, None))]
        self.f = self.dc.newFace(edges=self.hes)

    def tearDown(self):
        self.dc = None
        self.verts = None
        self.hes = None
        self.f = None

    #----------
    def test_face_creation(self):
        """ Test basic face construction """
        f = dcel.Face()
        self.assertIsNotNone(f)
        self.assertIsInstance(f, dcel.Face)

    def test_hull_creation(self):
        """ Test construction of a hull from a set of vertices """ 
        #A Set of Vertices:
        rawCoords = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        verts = [self.dc.newVertex(x) for x in rawCoords]
        hull, discard = dcel.Face.hull_from_vertices(verts)
        self.assertTrue(len(hull) == len(verts))

    def test_export(self):
        """ Test the export of a face """
        data = self.f._export()
        self.assertTrue(all([x in data for x in ['i','edges','sitex','sitey','data']]))
        self.assertEqual(len(data['edges']), 4)
        self.assertEqual(data['i'], self.f.index)

    def test_centroids(self):
        """ Test the calculation of a face's centroids """
        f = self.dc.newFace(coords=np.array([[1,0],[0,1],[-1,0],[0,-1]]))
        self.assertTrue(np.all(f.getAvgCentroid() == np.array([0,0])))

    def test_centroids_2(self):
        f = self.dc.newFace(coords=np.array([[2,0],[0,1],[-1,0],[0,-1]]))
        self.assertTrue(np.all(f.getAvgCentroid() == np.array([0.25,0])))

    def test_centroids_3(self):
        f = self.dc.newFace(coords=np.array([[10,10],[9,11],[8,10],[9,9]]))
        self.assertTrue(np.all(f.getAvgCentroid() == np.array([9,10])))

        
    def test_has_edges(self):
        """ Test whether a face has edges or not """
        self.assertTrue(self.f.has_edges())
        self.assertTrue(len(self.f.edgeList) > 0)
        emptyFace = self.dc.newFace()
        self.assertFalse(emptyFace.has_edges())
        self.assertTrue(len(emptyFace.edgeList) == 0)

    def test_edge_add(self):
        """ Test the addition of edges to the face """
        anEdge = self.dc.createEdge(np.array([0,0]), np.array([1,0]))
        aFace = self.dc.newFace()
        self.assertIsNone(anEdge.face)
        self.assertFalse(aFace.has_edges())
        aFace.add_edge(anEdge)
        self.assertTrue(aFace.has_edges())
        self.assertIsNotNone(anEdge.face)
        self.assertEqual(anEdge.face, aFace)
    
    def test_edge_removal(self):
        """ Test the removal of an edge from a face """
        anEdge = self.dc.createEdge(np.array([0,0]), np.array([1,0]))
        aFace = self.dc.newFace()
        self.assertFalse(aFace.has_edges())
        aFace.add_edge(anEdge)
        self.assertTrue(aFace.has_edges())
        aFace.remove_edge(anEdge)
        self.assertFalse(aFace.has_edges())
        self.assertIsNone(anEdge.face)
        
    def test_sort_edges(self):
        """ Test internally sorting the face's edges ccw """
        original = self.hes.copy()
        copiedEdges = self.hes.copy()
        shuffle(copiedEdges)
        aFace = self.dc.newFace()
        aFace.add_edges(copiedEdges)
        self.assertEqual(copiedEdges, aFace.edgeList)
        aFace.sort_edges()
        self.assertTrue(copiedEdges != aFace.edgeList or copiedEdges == self.hes)
        self.assertTrue(original == aFace.edgeList)

    def test_mark_for_cleanup(self):
        """ Test that a face can be marked for cleanup """
        self.assertFalse(self.f.markedForCleanup)
        self.f.markForCleanup()
        self.assertTrue(self.f.markedForCleanup)

    def test_subdivide(self):
        """ Test subdividing a face into two faces """
        f1, f2 = self.f.subdivide(self.f.edgeList[0], ratio=0.5, angle=90)
        self.assertEqual(f1, self.f)
        self.assertTrue(all([x.face==f1 for x in f1.edgeList]))
        self.assertTrue(all([x.face==f2 for x in f2.edgeList]))
        
    def test_add_vertex(self):
        """ Test adding a free vertex to a face """
        v1 = self.dc.newVertex(np.array([0,0]))
        emptyFace = self.dc.newFace()
        self.assertFalse(bool(emptyFace.free_vertices))
        emptyFace.add_vertex(v1)
        self.assertTrue(bool(emptyFace.free_vertices))
        
    def test_translate_edge_modified(self):
        """ Test modifying a face by moving an edge """
        original_coords = self.f.edgeList[0].toArray()
        f2, edit_e = self.f.translate_edge(np.array([1,0]), self.f.edgeList[0])
        self.assertEqual(edit_e, EditE.MODIFIED)
        self.assertEqual(self.f, f2)
        for e in f2.edgeList:
            self.assertTrue(np.allclose(e.twin.origin.loc, e.next.origin.loc))
        self.assertTrue(np.allclose(f2.edgeList[0].toArray(), original_coords + np.array([1,0])))
        
        
    def test_translate_edge_new(self):
        """ Test creating a new face from an existing one, by moving an edge """
        e = self.dc.createEdge(np.array([1,0]), np.array([4,0]))
        original_coords = self.f.edgeList[0].toArray()
        f2, edit_e = self.f.translate_edge(np.array([1,0]), self.f.edgeList[0])
        self.assertEqual(edit_e, EditE.NEW)
        self.assertNotEqual(self.f, f2)
        for edge in f2.edgeList:
            self.assertTrue(np.allclose(edge.twin.origin.loc, edge.next.origin.loc))
        self.assertTrue(np.allclose(f2.edgeList[0].toArray(), original_coords + np.array([1,0])))

    
    def test_translate_edge_force(self):
        """ Test forcing modification of a face by moving an edge """
        e = self.dc.createEdge(np.array([1,0]), np.array([4,0]))
        original_coords = self.f.edgeList[0].toArray()
        f2, edit_e = self.f.translate_edge(np.array([1,0]), self.f.edgeList[0], force=True)
        self.assertEqual(edit_e, EditE.MODIFIED)
        self.assertEqual(self.f, f2)
        for edge in f2.edgeList:
            self.assertTrue(np.allclose(edge.twin.origin.loc, edge.next.origin.loc))
        self.assertTrue(np.allclose(f2.edgeList[0].toArray(), original_coords + np.array([1,0])))

    
    def test_translate_edge_modified_by_candidates(self):
        """ Test modification of a face by designating non-constraining links """
        e = self.dc.createEdge(np.array([1,0]), np.array([4,0]))
        original_coords = self.f.edgeList[0].toArray()
        f2, edit_e = self.f.translate_edge(np.array([1,0]), self.f.edgeList[0], candidates=set([e]))
        self.assertEqual(edit_e, EditE.MODIFIED)
        self.assertEqual(self.f, f2)
        for edge in f2.edgeList:
            self.assertTrue(np.allclose(edge.twin.origin.loc, edge.next.origin.loc))
        self.assertTrue(np.allclose(f2.edgeList[0].toArray(), original_coords + np.array([1,0])))

        
        
    def test_merge_faces_initial(self):
        """ Test merging two faces into one, of the simplest case """
        f2 = self.dc.newFace()
        (mergedFace, discarded) = dcel.Face.merge_faces(self.f, f2)
        originalVerts = self.f.get_all_vertices()
        mergedVerts = mergedFace.get_all_vertices()
        self.assertEqual(originalVerts, mergedVerts)
        self.assertFalse(bool(discarded))

    def test_merge_faces_simple(self):
        """ Test merging two simple faces together """
        f1 = self.dc.newFace(coords=np.array([[-1,0],[1,0],[0,1]]))
        f2 = self.dc.newFace(coords=np.array([[0,0],[1,0],[0,-1]]))
        (mergedFace, discarded) = dcel.Face.merge_faces(f1, f2)
        originalVerts = f1.get_all_vertices().union(f2.get_all_vertices()).difference(discarded)
        mergedVerts = mergedFace.get_all_vertices()
        self.assertEqual(originalVerts, mergedVerts)
        self.assertTrue(len(discarded) == 1)
    
    def test_cut_out_modified(self):
        """ Test doing a no-op when a face is unconstrained """
        f1 = self.dc.newFace(coords=np.array([[2,0],[3,0],[2,4]]))
        original_coords = np.array([x.toArray() for x in f1.get_all_vertices()])
        original_indices = set([x.index for x in f1.get_all_vertices()])
        with self.dc:
            f2, edit_e = f1.cut_out()

        self.assertEqual(f1, f2)
        self.assertEqual(edit_e, EditE.MODIFIED)
        f2_coords = np.array([x.toArray() for x in f2.get_all_vertices()])
        original_coords.sort(axis=0)
        f2_coords.sort(axis=0)
        coords_equal = (original_coords == f2_coords).all()
        self.assertTrue(coords_equal)

        f2_indices = set([x.index for x in f2.get_all_vertices()])
        self.assertTrue(len(original_indices.intersection(f2_indices)) == 3)

    def test_cut_out_new(self):
        """ Test cloning a face, its halfedges, and its vertices """
        f1 = self.dc.newFace(coords=np.array([[2,0],[3,0],[2,4]]))
        e = f1.edgeList[0].extend(direction=np.array([1,1]), inSequence=False)
        original_coords = np.array([x.toArray() for x in f1.get_all_vertices()])
        original_indices = set([x.index for x in f1.get_all_vertices()])
        with self.dc:
            f2, edit_e = f1.cut_out()

        self.assertNotEqual(f1, f2)
        self.assertEqual(edit_e, EditE.NEW)
        f2_coords = np.array([x.toArray() for x in f2.get_all_vertices()])
        original_coords.sort(axis=0)
        f2_coords.sort(axis=0)
        coords_equal = (original_coords == f2_coords).all()
        self.assertTrue(coords_equal)

        f2_indices = set([x.index for x in f2.get_all_vertices()])
        self.assertTrue(len(original_indices.intersection(f2_indices)) == 0)

    def test_cut_out_modified_force(self):
        """ Test forcing a modification """
        f1 = self.dc.newFace(coords=np.array([[2,0],[3,0],[2,4]]))
        e = f1.edgeList[0].extend(direction=np.array([1,1]), inSequence=False)
        original_coords = np.array([x.toArray() for x in f1.get_all_vertices()])
        original_indices = set([x.index for x in f1.get_all_vertices()])
        with self.dc:
            f2, edit_e = f1.cut_out(force=True)

        self.assertEqual(f1, f2)
        self.assertEqual(edit_e, EditE.MODIFIED)
        f2_coords = np.array([x.toArray() for x in f2.get_all_vertices()])
        original_coords.sort(axis=0)
        f2_coords.sort(axis=0)
        coords_equal = (original_coords == f2_coords).all()
        self.assertTrue(coords_equal)

        f2_indices = set([x.index for x in f2.get_all_vertices()])
        self.assertTrue(len(original_indices.intersection(f2_indices)) == 3)

    def test_cut_out_modified_by_candidates(self):
        """ Test specifying non-constraining neighbours """
        f1 = self.dc.newFace(coords=np.array([[2,0],[3,0],[2,4]]))
        e = f1.edgeList[0].extend(direction=np.array([1,1]), inSequence=False)
        original_coords = np.array([x.toArray() for x in f1.get_all_vertices()])
        original_indices = set([x.index for x in f1.get_all_vertices()])
        candidate_set = set([f1, e.face, e.twin.face]).union(f1.edgeList).union([e, e.twin])
        with self.dc:
            f2, edit_e = f1.cut_out(candidates=candidate_set)

        self.assertEqual(edit_e, EditE.MODIFIED)
        self.assertEqual(f1, f2)

        f2_coords = np.array([x.toArray() for x in f2.get_all_vertices()])
        original_coords.sort(axis=0)
        f2_coords.sort(axis=0)
        coords_equal = (original_coords == f2_coords).all()
        self.assertTrue(coords_equal)

        f2_indices = set([x.index for x in f2.get_all_vertices()])
        self.assertTrue(len(original_indices.intersection(f2_indices)) == 3)

        
    def test_scale_modified(self):
        """ Test scaling a face """
        f2, edit_e = self.f.copy().scale(amnt=np.array([2,2]))
        self.assertEqual(edit_e, EditE.MODIFIED)
        for a,b in zip(self.f.edgeList, f2.edgeList):
            a_origin = a.origin.loc
            a_twin = a.twin.origin.loc
            b_origin = b.origin.loc
            b_twin = b.twin.origin.loc
            self.assertTrue(np.allclose(a_origin * 2, b_origin))
            self.assertTrue(np.allclose(a_twin * 2, b_twin))
        
    
    def test_scale_new(self):
        """ Test cloning a face then scaling it """
        e2 = self.dc.createEdge(np.array([1,0]), np.array([2,4]))
        f2, edit_e = self.f.scale(amnt=np.array([2,2]))
        self.assertEqual(edit_e, EditE.NEW)
        for a,b in zip(self.f.edgeList, f2.edgeList):
            a_origin = a.origin.loc
            a_twin = a.twin.origin.loc
            b_origin = b.origin.loc
            b_twin = b.twin.origin.loc
            self.assertTrue(np.allclose(a_origin * 2, b_origin))
            self.assertTrue(np.allclose(a_twin * 2, b_twin))


    def test_scale_force(self):
        """ Test forcing modification by scaling """
        original_locs = [x.toArray() for x in self.f.edgeList]
        e2 = self.dc.createEdge(np.array([1,0]), np.array([2,4]))
        f2, edit_e = self.f.scale(amnt=np.array([2,2]), force=True)
        self.assertEqual(edit_e, EditE.MODIFIED)
        for a,b in zip(original_locs, f2.edgeList):
            a_origin = a[0]
            a_twin = a[1]
            b_origin = b.origin.loc
            b_twin = b.twin.origin.loc
            self.assertTrue(np.allclose(a_origin * 2, b_origin))
            self.assertTrue(np.allclose(a_twin * 2, b_twin))
        self.assertTrue(np.allclose(e2.origin.loc, np.array([2,0])))

    def test_scale_modified_by_candidates(self):
        """ Test scaling as modification by specifying non-constraining connections """
        original_locs = [x.toArray() for x in self.f.edgeList]
        e2 = self.dc.createEdge(np.array([1,0]), np.array([2,4]))
        f2, edit_e = self.f.scale(amnt=np.array([2,2]), candidates=set([e2]))
        self.assertEqual(edit_e, EditE.MODIFIED)
        for a,b in zip(original_locs, f2.edgeList):
            a_origin = a[0]
            a_twin = a[1]
            b_origin = b.origin.loc
            b_twin = b.twin.origin.loc
            self.assertTrue(np.allclose(a_origin * 2, b_origin))
            self.assertTrue(np.allclose(a_twin * 2, b_twin))
        self.assertTrue(np.allclose(e2.origin.loc, np.array([2,0])))
    
    def test_rotate(self):
        """ Test rotating by modification """
        f1 = self.dc.newFace(coords=np.array([[2,0],[3,0],[2,-1]]))
        f2, edit_e1 = f1.rotate(radians(90))
        self.assertEqual(edit_e1, EditE.MODIFIED)
        self.assertEqual(f1, f2)
        allPoints = f2.get_all_coords()
        self.assertTrue((allPoints == np.array([2,1])).any(axis=1).any())
        f3, edit_e2 = f1.rotate(radians(180))
        self.assertEqual(f1, f3)
        self.assertEqual(edit_e2, EditE.MODIFIED)
        allPoints_2 = f3.get_all_coords()
        self.assertTrue((allPoints_2 == np.array([2,-1])).any(axis=1).any())

    def test_rotate_clockwise(self):
        """ Test rotating the other direction """
        f1 = self.dc.newFace(coords=np.array([[2,0],[3,0],[2,1]]))
        f2, edit_e = f1.rotate(radians(-90))
        self.assertEqual(edit_e, EditE.MODIFIED)
        allPoints = f2.get_all_coords()
        self.assertTrue((allPoints == np.array([2,-1])).any(axis=1).any())
        f3, edit_e2 = f1.rotate(radians(-180))
        self.assertEqual(edit_e2, EditE.MODIFIED)
        allPoints_2 = f3.get_all_coords()
        self.assertTrue((allPoints_2 == np.array([2,1])).any(axis=1).any())

    def test_has_constraints_false(self):
        """ A face isn't constrained by default """
        f = dcel.Face()
        self.assertFalse(f.has_constraints())

    def test_has_constraints_false_edges(self):
        """ A face isn't constrained with vertices and halfedges by default """
        f = self.dc.newFace(coords=np.array([[10,10],[11,10],[10,11]]))
        self.assertFalse(f.has_constraints())

    def test_has_constraints_false_abitrary(self):
        """ a face isn't constrained by arbitrary candidates """
        f = dcel.Face()
        self.assertFalse(f.has_constraints(set(["a"])))

    def test_has_constraints_false_face_candidate(self):
        """ a face isn't constrained by a passed in candidate face """
        f = self.dc.newFace(coords=np.array([[10,10],[11,10],[10,11]]))
        f2 = dcel.Face()
        f.edgeList[0].twin.face = f2
        self.assertFalse(f.has_constraints(set([f2])))

    def test_has_constraints_true_face(self):
        """ a face is constrained by a non-candidate face """
        logging.info("Starting test has constraints true face")
        f = self.dc.newFace(coords=np.array([[10,10],[11,10],[10,11]]))
        f2 = dcel.Face()
        f.edgeList[0].twin.face = f2
        self.assertTrue(f.has_constraints())

    def test_constrain_to_circle(self):
        central_loc = np.array([10,0])
        f = self.dc.newFace(coords=np.array([[10,0],[12,0],[10,2]]))
        original_verts = f.get_all_vertices()
        asArray = np.array([x.toArray() for x in original_verts])
        self.assertFalse((get_distance_raw(asArray, central_loc) <= 2).all())
        f2, edit_e = f.constrain_to_circle(central_loc, 1)
        self.assertEqual(edit_e, EditE.MODIFIED)
        self.assertEqual(f, f2)
        self.assertEqual(len(original_verts.symmetric_difference(f.get_all_vertices())), 0)
        self.assertEqual(len(f.edgeList), 2)
        new_verts = f.get_all_vertices()
        new_asArray = np.array([x.toArray() for x in new_verts])
        self.assertTrue((get_distance_raw(new_asArray, central_loc) <= 2).all())
    
    def test_fixup(self):
        e1 = self.dc.createEdge(np.array([0,0]), np.array([1,0]))
        e2 = self.dc.createEdge(np.array([1,0]), np.array([0,1]))
        f = self.dc.newFace(edges=[e1,e2])
        self.assertEqual(len(f.edgeList), 2)
        newEdges = f.fixup(self.dc.bbox)
        self.assertEqual(len(f.edgeList), 3)
        self.assertEqual(len(newEdges), 1)
        self.assertTrue((newEdges[0].toArray() == np.array([[0,1],[0,0]])).all())
        for a,b in zip(f.edgeList, islice(cycle(f.edgeList), 1, None)):
            self.assertEqual(a.next, b)
            self.assertEqual(b.prev, a)
            self.assertEqual(a.face, f)
        
        
if __name__ == "__main__":
      #use python $filename to use this logging setup
      LOGLEVEL = logging.INFO
      logFileName = "log.DCEL_FACE_Tests"
      logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
      console = logging.StreamHandler()
      console.setLevel(logging.INFO)
      logging.getLogger().addHandler(console)
      unittest.main()
      #reminder: user logging.getLogger().setLevel(logging.NOTSET) for log control
