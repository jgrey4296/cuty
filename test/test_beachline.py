import unittest
import logging
from test_context import cairo_utils as utils
from cairo_utils import beachline


class BeachLine_Tests(unittest.TestCase):

    def setUp(self):
        return 1

    def tearDown(self):
        return 1
        
    #--------------------
    # NODE TESTS
    #--------------------
    
    def test_node_creation(self):
        self.assertIsNotNone(beachline.NilNode)
        self.assertIsNotNone(beachline.Node)

        node = beachline.Node(5, act=False)

        self.assertIsNotNone(node)
        self.assertTrue(hasattr(node, "id"))
        self.assertTrue(node.red)
        self.assertFalse(node.arc)
        self.assertEqual(node.left, beachline.NilNode)
        self.assertEqual(node.right, beachline.NilNode)
        self.assertEqual(node.parent, beachline.NilNode)
        self.assertTrue(node.isLeaf())
        

    def test_node_arc_creation(self):
        return

    def test_node_comparison(self):
        return

    def test_node_arc_comparison(self):
        return

    def test_node_intersection(self):
        return

    def test_node_update_arc(self):
        return

    def test_node_getBlackHeight(self):
        return

    def test_node_countBlackHeight(self):
        return

    def test_node_get_predecessorAndSuccessor(self):
        return

    def test_node_minmax(self):
        return

    def test_node_add_leftright(self):
        return

    def test_node_disconnect_from_parent(self):
        return

    def test_node_link_leftright(self):
        return

    def test_node_disconnect_leftright(self):
        return

    #--------------------
    # BEACHLINE TESTS
    #--------------------

    def test_beachline_creation(self):
        return

    def test_beachline_isEmpty(self):
        return

    def test_beachline_insert(self):
        return

    def test_beachline_insert_many(self):
        return

    def test_beachline_insert_successor_predecessor(self):
        return

    def test_beachline_delete_value(self):
        return

    def test_beachline_delete_node(self):
        return

    def test_beachline_search(self):
        return

    def test_beachline_minmax(self):
        return

    def test_beachline_balance(self):
        return

    def test_beachline_get_chain(self):
        return

    def test_beachline_get_triples(self):
        return

    def test_beachline_countBlackHeight(self):
        return
          

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
