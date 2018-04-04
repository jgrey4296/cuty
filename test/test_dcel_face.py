import unittest
import logging
import numpy as np
import IPython
from random import shuffle
from itertools import islice, cycle
from test_context import cairo_utils as utils
from cairo_utils import dcel


class DCEL_FACE_Tests(unittest.TestCase):
    def setUp(self):
        self.dc = dcel.DCEL()
        self.verts = [self.dc.newVertex(x) for x in np.array([[1,0],[0,1],[-1,0],[0,-1]])]
        self.hes = [self.dc.newEdge(a,b) for a,b in zip(self.verts, islice(cycle(self.verts), 1, None))]
        self.f = self.dc.newFace(edges=self.hes)        

    def tearDown(self):
        self.dc = None
        self.verts = None
        self.hes = None
        self.f = None

    #----------
    def test_face_creation(self):
        f = dcel.Face()
        self.assertIsNotNone(f)
        self.assertIsInstance(f, dcel.Face)

    def test_hull_creation(self):
        #A Set of Vertices:
        rawCoords = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        verts = [self.dc.newVertex(x) for x in rawCoords]
        hull, discard = dcel.Face.hull_from_vertices(verts)
        assert(len(hull) == len(verts))

    def test_export(self):
        #create some vertices
        #create some halfedges
        #create the face

        #export it

        #verify
        return 0        

    def test_centroids(self):
        #create

        #avg centroid

        #centroid

        #bbox centroid
        
        return 0

    def test_has_edges(self):
        self.assertTrue(self.f.has_edges())
        self.assertTrue(len(self.f.edgeList) > 0)
        emptyFace = self.dc.newFace()
        self.assertFalse(emptyFace.has_edges())
        self.assertTrue(len(emptyFace.edgeList) == 0)

    def test_edge_add(self):
        anEdge = self.dc.createEdge(np.array([0,0]), np.array([1,0]))
        aFace = self.dc.newFace()
        self.assertIsNone(anEdge.face)
        self.assertFalse(aFace.has_edges())
        aFace.add_edge(anEdge)
        self.assertTrue(aFace.has_edges())
        self.assertIsNotNone(anEdge.face)
        self.assertEqual(anEdge.face, aFace)
    
    def test_edge_removal(self):
        anEdge = self.dc.createEdge(np.array([0,0]), np.array([1,0]))
        aFace = self.dc.newFace()
        self.assertFalse(aFace.has_edges())
        aFace.add_edge(anEdge)
        self.assertTrue(aFace.has_edges())
        aFace.remove_edge(anEdge)
        self.assertFalse(aFace.has_edges())
        self.assertIsNone(anEdge.face)
        
    def test_sort_edges(self):
        copiedEdges = self.hes.copy()
        shuffle(copiedEdges)
        aFace = self.dc.newFace()
        aFace.add_edges(copiedEdges)
        self.assertEqual(copiedEdges, aFace.edgeList)
        aFace.sort_edges()
        self.assertTrue(copiedEdges != aFace.edgeList or copiedEdges == self.hes)

    def test_mark_for_cleanup(self):
        self.assertFalse(self.f.markedForCleanup)
        self.f.markForCleanup()
        self.assertTrue(self.f.markedForCleanup)

    def test_subdivide(self):
        self.f.subdivide(self.f.edgeList[0], ratio=0.5, angle=90)

    def test_add_vertex(self):
        v1 = self.dc.newVertex(np.array([0,0]))
        emptyFace = self.dc.newFace()
        emptyFace.add_vertex(v1)

    def test_translate_edge(self):
        self.f.translate_edge(self.f.edgeList[0], np.array([1,0]))

    def test_merge_faces(self):
        f2 = self.dc.newFace()
        self.f.merge_faces(f2)
    
    def test_cut_out(self):
        self.f.cut_out()

    def test_scale(self):
        self.f.scale(amnt=2)

    def test_rotate(self):
        self.f.rotate(1)
    
    
if __name__ == "__main__":
      #use python $filename to use this logging setup
      LOGLEVEL = logging.INFO
      logFileName = "log.DCEL_FACE_Tests"
      logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
      console = logging.StreamHandler()
      console.setLevel(logging.WARN)
      logging.getLogger().addHandler(console)
      unittest.main()
      #reminder: user logging.getLogger().setLevel(logging.NOTSET) for log control
