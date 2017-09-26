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
        self.assertEqual(utils._author, "jgrey")

    #Test for specific constants?
    def test_main_loads(self):
        self.assertTrue(hasattr(utils, 'Quadratic'))
        self.assertTrue(hasattr(utils, 'Parabola'))
        self.assertTrue(hasattr(utils, 'Tree'))
        self.assertTrue(hasattr(utils, 'RBTree'))
        self.assertTrue(hasattr(utils, 'DCEL'))

    def test_has_math(self):;
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
