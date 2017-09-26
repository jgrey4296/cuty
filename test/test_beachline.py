import unittest
import logging
from test_context import cairo_utils as utils
from utils import beachline


class BeachLine_Tests(unittest.TestCase):

      def setUp(self):
            return 1

      def tearDown(self):
            return 1

      #----------
      #use testcase snippets

      #beachline creation
      #addition
      #removal
      #intersection

      #nilnode
      #node
      #operations
      

if __name__ == "__main__":
      #use python $filename to use this logging setup
      LOGLEVEL = logging.INFO
      logFileName = "log.beachline_tests"
      logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
      console = logging.StreamHandler()
      console.setLevel(logging.WARN)
      logging.getLogger().addHandler(console)
      unittest.main()
      #reminder: user logging.getLogger().setLevel(logging.NOTSET) for log control
