""" A General Line Intersector submodule """
#pylint: disable=too-few-public-methods
import heapq
import logging as root_logger
from functools import partial
from math import inf
import numpy as np

from ..heaputils import pop_while_same, HeapWrapper
from ..rbtree import RBTree, Directions
from ..constants import D_EPSILON
from .constants import SWEEP_NUDGE
from .vertex import Vertex
from .halfedge import HalfEdge


logging = root_logger.getLogger(__name__)
NEIGHBOUR_CONDITION = lambda v, n: n.value.contains_vertex(v)

class IntersectResult:
    """ Utility class to collect the start and end sets of halfedges,
    along with the set of intersections contained within halfedges,
    and the vertex describing that point.

    Starting and Ending are defined from Top -> Bottom, Left -> Right
    """

    def __init__(self, vert, start, contain, end):
        self.vertex = vert
        self.start = start
        self.contain = contain
        self.end = end

    def __repr__(self):
        return "Intersection: {}\nStart: {}\nContain: {}\nEnd: {}".format(self.vertex,
                                                                          self.start,
                                                                          self.contain,
                                                                          self.end)


#------------------------------
# def Comparison functions
#------------------------------

def line_cmp(a, b, cd):
    """ Line comparison to be used in the status tree """
    #Is horizontal:
    a_hor = a.value.is_flat()
    b_hor = b.is_flat()
    #---
    logging.debug("Comparison: {} - {}".format(a.value.index, b.index))
    logging.debug("Flat:       {} - {}".format(a_hor, b_hor))
    y = cd['y']
    if not (a_hor or b_hor):
        y += cd['nudge']
    a_ranges = a.value.get_ranges()
    b_ranges = b.get_ranges()

    a_val = a.value(y=y)[0]
    b_val = b(y=y)[0]
    if b_hor:
        b_val = min(max(cd['x'], b_ranges[0, 0]), b_ranges[0, 1])
    if a_hor:
        a_val = min(max(cd['x'], a_ranges[0, 0]), a_ranges[0, 1])

    logging.debug("Values: {} - {}".format(a_val, b_val))

    if a_val <= b_val:
        return Directions.RIGHT
    return Directions.LEFT

def line_eq(a, b, cd):
    """ Compare two lines against each other """
    #pylint: disable=unused-argument
    return a.value == b

def line_cmp_vert(a, b, cd):
    """ Compare a line against a point """
    result = Directions.LEFT
    if a.value.is_flat():
        a_val = cd['x']
    else:
        a_val = a.value(y=cd['y'])[0]
    logging.debug("VERT a_val{}: {}  b_val: {}".format(a.value.index, a_val, b[0]))

    if a_val <= b[0]:
        result = Directions.RIGHT
    return result

def line_eq_vert(a, b, cd):
    """ test for equality between a line and a point """
    return np.allclose(a.value(y=cd['y']), b)


#------------------------------
# def MAIN CLASS
#------------------------------

class LineIntersector:
    """ Processes a DCEL to intersect half_edges,
    in a self contained class
    """

    def __init__(self, dcel):
        self.dcel = dcel
        self.edge_set = set()
        self.lower_edges = []
        self.results = []
        self.discovered = set()
        self.sweep_y = inf
        self.sweep_y_prev = inf
        #Tree to keep active edges in,
        #Sorted by the x's for the current y of the sweep line
        self.status_tree = RBTree(cmp_func=line_cmp,
                                  eq_func=line_eq)
        #Heap of (vert, edge) pairs,
        #with invariant: all([e.is_upper() for v, e in event_list])
        self.event_list = []


    #------------------------------
    # def MAIN CALL
    #------------------------------

    def __call__(self, edge_set=None):
        #pylint: disable=unused-argument
        self.initialise_data(edge_set=edge_set)
        assert(bool(self.event_list))
        assert(bool(self.discovered))
        assert(not bool(self.results))
        assert(self.sweep_y == inf)

        #Main loop of the Intersection Algorithm
        while bool(self.event_list):
            logging.debug("--------------------")
            logging.debug("Event list remaining: {}".format(len(self.event_list)))
            logging.debug("\n".join([repr(x) for x in self.event_list]))

            curr_vert, curr_edge_list = self.get_next_event()
            logging.debug("Curr Vert: {}".format(curr_vert))
            self.update_sweep(curr_vert)
            self.debug_chain("Initial")

            logging.debug("Searching tree")
            closest_node, _ = self.search_tree(curr_vert)
            upper_set, contain_set, lower_set = self.determine_sets(curr_vert,
                                                                    closest_node,
                                                                    curr_edge_list.copy())
            self.report_intersections(curr_vert, upper_set, contain_set, lower_set)

            #todo: delete non-flat points of the non-flat event
            self.delete_values(contain_set, curr_x=curr_vert.loc[0])
            self.delete_values(lower_set, curr_x=curr_vert.loc[0])
            self.debug_chain("Deleted")


            #insert the segments with the status line a little lower
            candidate_lines = contain_set.union(upper_set)
            new_nodes = self.insert_values(candidate_lines, curr_x=curr_vert.loc[0])

            #Calculate additional events
            self.debug_chain("Inserted")

            self.handle_new_events(curr_vert, new_nodes)

        assert(self.sweep_y != inf)
        assert(not bool(self.event_list))
        return self.results

    def initialise_data(self, edge_set=None):
        """ Setup the internal structures to run a line sweep algorithm """
        logging.debug("Starting line intersection algorithm")
        self.results = []
        #setup the set of edges to intersect
        if edge_set is None:
            self.edge_set = self.dcel.half_edges.copy()
        else:
            assert(isinstance(edge_set, set))
            #get the twins as well
            twins = [x.twin for x in edge_set]
            edge_set.update(twins)
            self.edge_set = edge_set
        assert(self.edge_set is not None)
        self.lower_edges = [e for e in self.edge_set if not e.is_upper()]
        self.event_list = [HeapWrapper(x.origin, x, desc="initial") \
                           for x in self.edge_set.difference(self.lower_edges)]
        self.event_list += [HeapWrapper(x.origin, x.twin, desc="initial_twin") \
                            for x in self.lower_edges]
        heapq.heapify(self.event_list)
        self.discovered.update([x.ordinal for x in self.event_list])

        logging.debug("EdgeSet: {}, lower_edges: {}".format(len(self.edge_set),
                                                            len(self.lower_edges)))
        logging.debug("Event_list: {}".format(len(self.event_list)))

    def determine_sets(self, curr_vert, closest, candidates):
        """ TODO: describe """
        #pylint: disable=no-self-use
        assert(isinstance(curr_vert, Vertex))
        upper_set = set()
        contain_set = set()
        lower_set = set()

        if closest is not None:
            closest_segment = closest.value
            #if a line is horizontal, switch to a call(y) while,
            #and get lines until not in horizontal bounds
            candidate_nodes = closest.get_neighbours_while(partial(NEIGHBOUR_CONDITION, curr_vert))
            candidates += [x.value for x in candidate_nodes]
            candidates.append(closest_segment)

        logging.debug("Candidates: {}".format(candidates))
        #todo: adapt for horizontal lines
        #init a dictionary for intersections for each individual horizontal line
        #foreach individual horizontal line, test candidates until failure, then break
        for c in candidates:
            if c.origin == curr_vert:
                upper_set.add(c)
            elif c.twin.origin == curr_vert:
                lower_set.add(c)
            elif c.contains_vertex(curr_vert):
                contain_set.add(c)
        logging.debug("Upper Set: {}".format([x.index for x in upper_set]))
        logging.debug("Contain Set: {}".format([x.index for x in contain_set]))
        logging.debug("Lower Set: {}".format([x.index for x in lower_set]))
        return (upper_set, contain_set, lower_set)

    def handle_new_events(self, curr_vert, new_nodes):
        """ Given new nodes to add, process them """
        #pylint: disable=unused-argument
        logging.debug("Finding new Events")
        #todo: only find new events if there is a set of non-horizontal lines
        if not bool(new_nodes):
            closest_node, _ = self.search_tree(curr_vert)
            if closest_node is None:
                return
            left_n = closest_node.get_predecessor()
            right_n = closest_node.get_successor()
            if left_n is not None:
                self.find_new_events(left_n.value, closest_node.value, curr_vert.to_array())
            if right_n is not None:
                self.find_new_events(closest_node.value, right_n.value, curr_vert.to_array())

        else:
            logging.debug("New nodes added")
            assert(bool(self.status_tree))
            if not bool(new_nodes):
                return
            #TODO: this could be more efficient
            chain = self.status_tree.get_chain()
            ordinalered = [x for x in chain if x in new_nodes]
            logging.debug("New Nodes: {}".format([x.value.index for x in ordinalered]))
            leftmost = ordinalered[0]
            leftmost_n = leftmost.get_predecessor()
            if leftmost_n is not None and leftmost is not None:
                self.find_new_events(leftmost_n.value,
                                     leftmost.value,
                                     curr_vert.to_array())


            rightmost = ordinalered[-1]
            rightmost_n = rightmost.get_successor()

            if rightmost is not None and rightmost_n is not None:
                self.find_new_events(rightmost.value,
                                     rightmost_n.value,
                                     curr_vert.to_array())

    def find_new_events(self, a, b, loc):
        """ Given two edges and a location of the sweep line,
        find any valid intersection
        """
        assert(isinstance(loc, np.ndarray))
        assert(isinstance(a, HalfEdge))
        assert(isinstance(b, HalfEdge))

        logging.debug("Finding Events for: {}, {}, {}".format(a.index, b.index, loc))
        intersection = a.intersect(b)
        match_vert = None
        if intersection is None:
            logging.debug("No intersection")
            return
        if a.is_flat() or b.is_flat():
            logging.debug("Forcing same y position")
            intersection[1] = loc[1]

        #TODO: only works on cartesian
        if intersection[1] > loc[1]:
            logging.debug("Intersection too high: {} -- {}".format(intersection[1], loc[1]))
            return
        if intersection[1] < loc[1] or\
           (intersection[1] == loc[1] and loc[0] <= intersection[0]):
            logging.debug("Within bounds")
            match_vert = self.dcel.new_vertex(intersection)
            if match_vert in self.discovered:
                logging.debug("Vertex already discovered")
                return
        #finally, success!:
        if match_vert is not None:
            self.discovered.add(match_vert)
            wrapped = HeapWrapper(match_vert, a, desc="newEvent")
            wrapped2 = HeapWrapper(match_vert, b, desc="newEvent")
            logging.debug("Adding: {}".format(wrapped))
            heapq.heappush(self.event_list, wrapped)
            heapq.heappush(self.event_list, wrapped2)


    #------------------------------
    # def UTILITIES
    #------------------------------

    def debug_chain(self, the_str=None):
        """ log the flattened structure of the internal status tree """
        if the_str is None:
            the_str = ""
        chain = [x.value.index for x in self.status_tree.get_chain()]
        logging.debug("{} Tree Chain is: {}".format(the_str, chain))
        encountered = set()
        for x in chain:
            if x in encountered:
                raise Exception("Duplicated: {}".format(x))
            encountered.add(x)

    def report_intersections(self, v, u, c, l):
        """ Add intersection results when applicable """
        #todo: for each horizontal crossing point separately
        if sum([len(x) for x in [u, l, c]]) > 1:
            logging.debug("Reporting intersections")
            self.results.append(IntersectResult(v, u, c, l))

    def get_next_event(self):
        """ Get all segments that are on the current vertex """
        curr_vert, curr_edge_list = pop_while_same(self.event_list)
        assert(not bool(curr_edge_list) or all([isinstance(x, HalfEdge) for x in curr_edge_list]))
        return curr_vert, curr_edge_list

    def delete_values(self, values, curr_x):
        """ Remove a value from the status tree  """
        logging.debug("--------------------")
        logging.debug("Deleting values: {}".format([x.index for x in values]))
        assert(all([x.is_upper() for x in values]))
        chain = [x.value.index for x in self.status_tree.get_chain()]
        logging.debug("Chain: {}".format(chain))
        self.status_tree.delete_value(*values, cmp_data={'y':self.sweep_y, 'nudge': -SWEEP_NUDGE,
                                                         'x': curr_x - D_EPSILON})


        #Verify:
        chain = [x.value.index for x in self.status_tree.get_chain()]
        assert(all([x.index not in chain for x in values]))

    def insert_values(self, candidates, curr_x):
        """ Insert values into the status tree """
        logging.debug("Inserting values: {}".format([x.index for x in candidates]))
        flat_lines = set([x for x in candidates if x.is_flat()])
        #insert only non-flat lines
        to_add = candidates.difference(flat_lines)
        assert(all([x.is_upper() for x in to_add]))
        new_nodes = self.status_tree.insert(*to_add,
                                            cmp_data={'y':self.sweep_y, 'nudge': SWEEP_NUDGE,
                                                      'x': curr_x + D_EPSILON})
        new_nodes += self.status_tree.insert(*flat_lines,
                                             cmp_data={'y':self.sweep_y, 'nudge':SWEEP_NUDGE,
                                                       'x': curr_x + D_EPSILON})

        return new_nodes

    def update_sweep(self, curr):
        """ Increment the position of the frontier """
        assert(isinstance(curr, Vertex))
        candidate = curr.to_array()[1]
        if candidate > self.sweep_y:
            raise Exception("Sweep line moved in wrong direction: {} vs {}".format(candidate,
                                                                                   self.sweep_y))
        self.sweep_y = candidate

    def search_tree(self, curr):
        """ Search the status tree for a vertex """
        assert(isinstance(curr, Vertex))
        closest_node, d = self.status_tree.search(curr.to_array(),
                                                  cmp_data={'y': self.sweep_y,
                                                            'x': curr.to_array()[0]},
                                                  closest=True,
                                                  cmp_func=line_cmp_vert,
                                                  eq_func=line_eq_vert)

        if closest_node is not None:
            logging.debug("Closest Node: {}".format(closest_node.value.index))
        else:
            logging.debug("Closest Node is None")
        return closest_node, d
