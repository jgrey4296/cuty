import unittest
import logging
from test_context import cairo_utils as utils
from utils import dcel


class DCEL_Tests(unittest.TestCase):

      def setUp(self):
            return 1

      def tearDown(self):
            return 1

      #----------
      #use testcase snippets

      #test dcel creation
      
      #test Vertex

      #test halfedge

      #est Line

      #test Face

      #test dcel

      ##export
      ##import
      ##loadfile
      
      ##new Vertex/Edge/Face

      ##linkEdges

      ##setFaceForEdgeLoop

      ##orderVertices

      ##constrainhalf_edges

      ##purge infinite/edges/vertices/faces

      ##complete_faces

      ##create_corner_vertex

      ##fixup_halfedges

      ##verify_edges

      ##verify_faces_and_edges

      ##constrain_to_circle
      

if __name__ == "__main__":
      #use python $filename to use this logging setup
      LOGLEVEL = logging.INFO
      logFileName = "log.DCEL_Tests"
      logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
      console = logging.StreamHandler()
      console.setLevel(logging.WARN)
      logging.getLogger().addHandler(console)
      unittest.main()
      #reminder: user logging.getLogger().setLevel(logging.NOTSET) for log control
