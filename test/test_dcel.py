from test_context import cairo_utils as utils
from cairo_utils import dcel
import unittest
import logging
import IPython
#Import TestCase Instances here:
from test_dcel_vertex import DCEL_VERTEX_Tests
from test_dcel_halfedge import DCEL_HALFEDGE_Tests
from test_dcel_face import DCEL_FACE_Tests
from test_dcel_actual import DCEL_ACTUAL_Tests
from test_dcel_line_intersect import DCEL_LINE_INTERSECT_Tests
##

if __name__ == '__main__':
    #use python $filename to use this logging setup
    LOGLEVEL = logging.INFO
    logFileName = "log.dcel_comprehensive_tests"
    logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.WARN)
    logging.getLogger().addHandler(console)

    unittest.main()
