import unittest
import logging
import IPython
from test_context import cairo_utils as utils
from cairo_utils import rbtree
from cairo_utils.rbtree import ComparisonFunctions as CompFuncs
from cairo_utils.rbtree import Node


class RBTree_Tests(unittest.TestCase):

    def setUp(self):
        self.n = Node(2)

    def tearDown(self):
        self.n = None

    def test_creation(self):
        self.assertIsNotNone(self.n)
        self.assertIsInstance(self.n, Node)
        self.assertIsNone(self.n.left)
        self.assertIsNone(self.n.right)
        self.assertIsNone(self.n.parent)
        self.assertTrue(self.n.red)

    def test_eq(self):
        self.assertTrue(self.n == self.n)
        other = Node(2)
        self.assertFalse(self.n == other)
        other2 = Node(5)
        self.assertFalse(self.n == other2)

    def test_blackheight(self):
        self.assertEqual(self.n.getBlackHeight(), 0)

    def test_blackheight_2(self):
        n1 = Node(1)
        n2 = Node(2, n1)
        n3 = Node(3, n2)
        self.assertEqual(n3.getBlackHeight(), 0)
        n3.red = False
        self.assertEqual(n3.getBlackHeight(), 1)
        n2.red = False
        self.assertEqual(n3.getBlackHeight(), 2)
        n1.red = False
        self.assertEqual(n3.getBlackHeight(), 3)
    
    def test_link_left(self):
        n1 = Node(2)
        n2 = Node(3)
        self.assertIsNone(n1.left)
        self.assertIsNone(n1.right)
        n1.link_left(n2)
        self.assertEqual(n1.left, n2)
        self.assertIsNone(n1.right)
        self.assertEqual(n2.parent, n1)

    def test_link_right(self):
        n1 = Node(2)
        n2 = Node(3)
        self.assertIsNone(n1.left)
        self.assertIsNone(n1.right)
        n1.link_right(n2)
        self.assertEqual(n1.right, n2)
        self.assertIsNone(n1.left)
        self.assertEqual(n2.parent, n1)

    def test_link_left_fails(self):
        n1 = Node(2)
        with self.assertRaises(Exception):
            n1.link_left(n1)
        n2 = Node(5)
        n1.link_left(n2)
        with self.assertRaises(Exception):
            n2.link_left(n1)
            
    def test_link_right_fails(self):
        n1 = Node(5)
        with self.assertRaises(Exception):
            n1.link_right(n1)
        n2 = Node(10)
        n1.link_right(n2)
        with self.assertRaises(Exception):
            n2.link_right(n1)

    def test_add_left(self):
        n1 = Node(2)
        n2 = Node(3)
        self.assertIsNone(n1.left)
        n1.add_left(n2)
        self.assertEqual(n1.left, n2)

    def test_add_right(self):
        n1 = Node(2)
        n2 = Node(3)
        self.assertIsNone(n1.right)
        n1.add_right(n2)
        self.assertEqual(n1.right, n2)

    def test_get_predecessor_None(self):
        n1 = Node(2)
        self.assertIsNone(n1.getPredecessor())

    def test_get_predecessor_basic(self):
        n1 = Node(2)
        n2 = Node(3)
        n1.add_left(n2)
        self.assertEqual(n1.getPredecessor(), n2)
        self.assertIsNone(n2.getPredecessor())
        
    def test_get_predecessor_basic_2(self):
        n1 = Node(2)
        n2 = Node(3)
        n1.add_right(n2)
        self.assertEqual(n2.getPredecessor(), n1)

    def test_get_predecessor(self):
        n1 = Node(2)
        n2 = Node(3)
        n3 = Node(4)
        n1.add_right(n2)
        n2.add_left(n3)
        self.assertEqual(n3.getPredecessor(), n1)

    def test_get_successor_None(self):
        n1 = Node(2)
        self.assertIsNone(n1.getSuccessor())

    def test_get_successor_basic(self):
        n1 = Node(2)
        n2 = Node(3)
        n1.add_left(n2)
        self.assertEqual(n2.getSuccessor(), n1)

    def test_get_successor_basic_2(self):
        n1 = Node(2)
        n2 = Node(3)
        n1.add_right(n2)
        self.assertEqual(n1.getSuccessor(), n2)

    def test_get_successor(self):
        n1 = Node(2)
        n2 = Node(3)
        n3 = Node(4)
        n1.add_left(n2)
        n2.add_right(n3)
        self.assertEqual(n3.getSuccessor(), n1)

    #test getPred/Succ_while

    def test_min(self):
        n1 = Node(2)
        n2 = Node(3)
        n3 = Node(4)
        n1.add_left(n2)
        n2.add_left(n3)
        self.assertEqual(n1.min(), n3)

    def test_max(self):
        n1 = Node(2)
        n2 = Node(3)
        n3 = Node(4)
        n1.add_right(n2)
        n2.add_right(n3)
        self.assertEqual(n1.max(), n3)
        
    def test_disconnect_from_parent(self):
        n1 = Node(2)
        n2 = Node(3)
        n1.add_left(n2)
        self.assertEqual(n2.parent, n1)
        self.assertEqual(n1.left, n2)
        n2.disconnect_from_parent()
        self.assertIsNone(n2.parent)
        self.assertIsNone(n1.left)

    def test_disconnect_left(self):
        n1 = Node(2)
        n2 = Node(3)
        n1.add_left(n2)
        self.assertEqual(n2.parent, n1)
        self.assertEqual(n1.left, n2)
        n1.disconnect_left()
        self.assertIsNone(n2.parent)
        self.assertIsNone(n1.left)
        
    def test_disconnect_right(self):
        n1 = Node(2)
        n2 = Node(3)
        n1.add_right(n2)
        self.assertEqual(n2.parent, n1)
        self.assertEqual(n1.right, n2)
        n1.disconnect_right()
        self.assertIsNone(n2.parent)
        self.assertIsNone(n1.right)
        
    def test_on_left(self):
        n1 = Node(2)
        n2 = Node(3)
        n3 = Node(4)
        n1.add_left(n2)
        n1.add_right(n3)
        self.assertTrue(n1.on_left(n2))
        self.assertFalse(n1.on_left(n3))

    def test_rotate_right(self):
        n0 = Node(1)
        n1 = Node(2)
        n2 = Node(3)
        n3 = Node(4)
        n4 = Node(5)
        n0.add_left(n1)
        n1.add_left(n2)
        n1.add_right(n3)
        n2.add_right(n4)
        setAsRoot, newHead = n1.rotate_right()
        #self.assertTrue(setAsRoot)
        self.assertEqual(newHead, n2)
        self.assertEqual(n1.left, n4)
        self.assertEqual(n4.parent, n1)
        self.assertEqual(n0.left, newHead)

    def test_rotate_left(self):
        n0 = Node(1)
        n1 = Node(2)
        n2 = Node(3)
        n3 = Node(4)
        n4 = Node(5)
        n0.add_right(n1)
        n1.add_left(n2)
        n1.add_right(n3)
        n3.add_left(n4)
        setAsRoot, newHead = n1.rotate_left()
        #self.assertTrue(setAsRoot)
        self.assertEqual(newHead, n3)
        self.assertEqual(n1.right, n4)
        self.assertEqual(n4.parent, n1)
        self.assertEqual(n0.right, newHead)
        
    def test_get_pred_succ_while_no_condition(self):
        n1 = Node(2)
        n2 = Node(3)
        n3 = Node(4)
        n4 = Node(5)
        n1.add_left(n2)
        n1.add_right(n3)
        n3.add_left(n4)
        succresults = n2.getSuccessor_while(lambda x: True)
        self.assertEqual(succresults, [n1,n4,n3])
        predresults = n3.getPredecessor_while(lambda x: True)
        self.assertEqual(predresults, [n4,n1,n2])
        neighbours = n1.getNeighbours_while(lambda x: True)
        self.assertEqual(neighbours, [n2,n4,n3])

    def test_is_leaf(self):
        n1 = Node(2)
        self.assertTrue(n1.isLeaf())
        n1.add_left(Node(34))
        self.assertFalse(n1.isLeaf())
        n1.add_right(Node(5))
        self.assertFalse(n1.isLeaf())
        
    #----------
        
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
