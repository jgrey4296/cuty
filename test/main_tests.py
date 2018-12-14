import unittest
import logging
from test_context import cairo_utils as utils

class Cairo_Util_Tests(unittest.TestCase):

    def setUp(self):
        return 1

    def tearDown(self):
        return 1

    #----------
    #use testcase snippets
    def test_author(self):
        self.assertEqual(utils._AUTHOR, "jgrey")

    #Test for specific constants?
    def test_main_loads(self):
        self.assertTrue(hasattr(utils, 'quadratic'))
        self.assertTrue(hasattr(utils, 'parabola'))
        self.assertTrue(hasattr(utils, 'tree'))
        self.assertTrue(hasattr(utils, 'rbtree'))
        self.assertTrue(hasattr(utils, 'dcel'))
        self.assertTrue(hasattr(utils, 'make_gif'))
        self.assertTrue(hasattr(utils, 'math'))
        
        
if __name__ == "__main__":
      #use python $filename to use this logging setup
      LOGLEVEL = logging.INFO
      logFileName = "log.Cairo_Util_Tests"
      logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
      console = logging.StreamHandler()
      console.setLevel(logging.WARN)
      logging.getLogger().addHandler(console)
      unittest.main()
      #reminder: user logging.getLogger().setLevel(logging.NOTSET) for log control
