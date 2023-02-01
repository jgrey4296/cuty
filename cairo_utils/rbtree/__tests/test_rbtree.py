import unittest
import logging
from test_context import cairo_utils as utils
from cairo_utils import rbtree
from cairo_utils.rbtree import comparison_functions as CompFuncs


class RBTree_Tests(unittest.TestCase):

    def setUp(self):
        self.t = rbtree.RBTree()

    def tearDown(self):
        self.t = None

    #----------
    #creation
    def test_creation(self):
        self.assertIsNotNone(self.t)
        self.assertIsInstance(self.t, rbtree.RBTree)

    #empty
    def test_empty(self):
        self.assertEqual(len(self.t), 0)
        self.assertFalse(bool(self.t))

    #insert
    def test_insert_empty(self):
        self.t.insert(2)
        self.assertEqual(len(self.t), 1)
        self.assertTrue(bool(self.t))
        self.t.insert(3)
        self.assertEqual(len(self.t), 2)

    #min
    def test_min(self):
        self.t.insert(4,2,6,5,2,7,8,4,2,5,2,1)
        self.assertEqual(len(self.t),12)
        m = self.t.min()
        self.assertIsInstance(m, rbtree.Node)
        self.assertEqual(m.value, 1)

    #max
    def test_max(self):
        self.t.insert(4,2,6,5,2,7,8,4,2,5,2,1)
        self.assertEqual(len(self.t),12)
        m = self.t.max()
        self.assertIsInstance(m, rbtree.Node)
        self.assertEqual(m.value, 8)

    def test_cmp(self):
        """ Swaps the ordering using a custom cmp function """
        self.t.cmpFunc = CompFuncs.inverted_comparison
        self.t.insert(4,2,6,5,2,7,8,4,2,5,2,1)
        self.assertEqual(len(self.t),12)
        mi = self.t.min()
        ma = self.t.max()
        self.assertIsInstance(mi, rbtree.Node)
        self.assertIsInstance(ma, rbtree.Node)
        self.assertEqual(mi.value, 1)
        self.assertEqual(ma.value, 8)

    #delete
    def test_delete(self):
        self.t.insert(4,2,6,5,2,7,8,4,2,5,2,1)
        self.assertEqual(len(self.t),12)
        m = self.t.max()
        self.assertIsInstance(m, rbtree.Node)
        self.assertEqual(m.value, 8)
        self.t.delete(m)
        self.assertEqual(len(self.t), 11)

    #search
    def test_search(self):
        self.t.insert(4,2,6,5,2,7,8,4,2,5,2,1)
        self.assertEqual(len(self.t),12)
        found, side = self.t.search(4)
        self.assertIsNotNone(found)
        self.assertIsInstance(found, rbtree.Node)

    def test_search_missing(self):
        self.t.insert(4,2,6,5,2,7,8,4,2,5,2,1)
        self.assertEqual(len(self.t),12)
        found, side = self.t.search(55)
        self.assertIsNone(found)

    def test_get_chain(self):
        baseList = [4,2,6,5,2,7,8,4,2,5,2,1]
        self.t.insert(*baseList)
        self.assertEqual(len(self.t),12)
        chain = self.t.get_chain()
        self.assertIsInstance(chain, list)
        self.assertTrue(all([isinstance(x, rbtree.Node) for x in chain]))
        values = [x.value for x in chain]
        self.assertEqual(values, sorted(baseList))

    def test_colours(self):
        baseList = [4,2,6,5,2,7,8,4,2,5,2,1]
        self.t.insert(*baseList)
        self.assertEqual(len(self.t),12)
        chain = self.t.get_chain()
        for node in chain:
            if node.red:
                self.assertTrue(node.left is None or not node.left.red)
                self.assertTrue(node.right is None or not node.right.red)
        #todo: check it is maintained when inserting and deleting

    #test that the path to farthest leaf is no more than
    #twice as long as path to shortest

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
