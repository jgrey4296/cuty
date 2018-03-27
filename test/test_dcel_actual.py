import unittest
import logging
from test_context import cairo_utils as utils
from cairo_utils import dcel


class DCEL_ACTUAL_Tests(unittest.TestCase):
    def setUp(self):
        self.dc = dcel.DCEL()
        return 1

    def tearDown(self):
        self.dc = None
        return 1

    #----------
    #test dcel creation
    def test_dcel_creation(self):
        self.assertIsInstance(self.dc, dcel.DCEL)

    def test_frontier(self):
        return 0

    def test_copy(self):
        return 0

    def test_export_import(self):
        return 0

    def test_quad_tree(self):
        return 0

    def test_vertex_creation(self):
        return 0

    def test_halfedge_creation(self):
        return 0

    def test_face_creation(self):
        return 0

    def test_path_creation(self):
        return 0

    def test_edge_linking(self):
        return 0

    def test_face_edge_loop(self):
        return 0

    def test_vertex_ordering(self):
        return 0

    def test_constraints(self):
        return 0

    def test_purging(self):
        return 0

    def test_complete_faces(self):
        return 0

    def test_edge_connections(self):
        return 0

    def test_corner_vertices(self):
        return 0

    def test_halfedge_fixup(self):
        return 0

    def test_verification(self):
        return 0

        
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
