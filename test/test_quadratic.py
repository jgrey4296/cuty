import unittest
import logging
import IPython
import numpy as np
from test_context import cairo_utils as utils
from cairo_utils import Quadratic


class Quadratic_Tests(unittest.TestCase):

      def setUp(self):
          self.q = Quadratic(1,2,3)
          self.q2 = Quadratic(1,-2,0)
          self.q3 = Quadratic(3,4,-2)

      def tearDown(self):
          self.q = None

      #----------
      #creation
      def test_creation(self):
          self.assertIsNotNone(self.q)
          self.assertIsInstance(self.q, Quadratic)

      #call
      def test_call(self):
          result = self.q(2)
          self.assertEqual(result, 11)

      #intersect
      def test_intersect(self):
          intersections = self.q2.intersect(self.q3)
          self.assertIsInstance(intersections, np.ndarray)
          self.assertEqual(len(intersections), 2)
          scaled = np.array(intersections * 1000, dtype=np.int)
          self.assertTrue(np.allclose(scaled, np.array([-3302, 302])))

      #discriminant

      #solve


if __name__ == "__main__":
      #use python $filename to use this logging setup
      LOGLEVEL = logging.INFO
      logFileName = "log.Quadratic_Tests"
      logging.basicConfig(filename=logFileName, level=LOGLEVEL, filemode='w')
      console = logging.StreamHandler()
      console.setLevel(logging.WARN)
      logging.getLogger().addHandler(console)
      unittest.main()
      #reminder: user logging.getLogger().setLevel(logging.NOTSET) for log control
