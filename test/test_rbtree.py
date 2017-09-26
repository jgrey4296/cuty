import unittest
import logging
from test_context import cairo_utils as utils
from utils import rbtree


class RBTree_Tests(unittest.TestCase):

      def setUp(self):
            return 1

      def tearDown(self):
            return 1

      #----------
      #creation

      #insert

      #delete

      #search

      #min

      #max

      #countBlackHeight
      

if __name__ == "__main__":
      #use python $filename to use this logging setup
      LOGLEVEL = logging.INFO
      logFileName = "log.rbtree_tests"
      logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
      console = logging.StreamHandler()
      console.setLevel(logging.WARN)
      logging.getLogger().addHandler(console)
      unittest.main()
      #reminder: user logging.getLogger().setLevel(logging.NOTSET) for log control
