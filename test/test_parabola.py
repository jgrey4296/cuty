import unittest
import logging
from test_context import cairo_utils as utils
from utils import Parabola


class Parabola_Tests(unittest.TestCase):

      def setUp(self):
            return 1

      def tearDown(self):
            return 1

      #----------
      #creation

      #str

      #is_left_of_focus

      #update_d

      #intersect

      #calcStandardForm

      #calcVertexForm

      #calc

      #call

      #to_numpy_array

      #eq

      #get_focus

      #toStandardForm

      #toVertexForm
      

if __name__ == "__main__":
      #use python $filename to use this logging setup
      LOGLEVEL = logging.INFO
      logFileName = "log.Parabola_Tests"
      logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
      console = logging.StreamHandler()
      console.setLevel(logging.WARN)
      logging.getLogger().addHandler(console)
      unittest.main()
      #reminder: user logging.getLogger().setLevel(logging.NOTSET) for log control
