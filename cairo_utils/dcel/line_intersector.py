import heapq
import logging as root_logger
from functools import partial
import IPython
import numpy as np
from collections import namedtuple

from ..heaputils import pop_while_same, HeapWrapper
from ..rbtree import RBTree
from .constants import SWEEP_NUDGE
from .Vertex import Vertex
from .HalfEdge import HalfEdge


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
        

class LineIntersector:
    """ Processes a DCEL to intersect halfEdges, 
    in a self contained class
    """

    def __new__(self, edgeSet=None, dcel=None):
        logging.debug("Starting line intersection algorithm")
        #setup the set of edges to intersect
        assert(edgeSet is not None or dcel is not None)
        if edgeSet is None:
            edgeSet = dcel.halfEdges.copy()
        else:
            assert(isinstance(edgeSet, set))
            edgeSet = edgeSet
        lowerEdges = [e for e in edgeSet if not e.isUpper()]
        
        logging.debug("EdgeSet: {}, lowerEdges: {}".format(len(edgeSet), len(lowerEdges)))
        results = []
        discovered = set()
        #Tree to keep active edges in,
        #Sorted by the x's for the current y of the sweep line
        status_tree = RBTree(cmpFunc=lambda a,b,cd: a.value(y=cd)[0] < b(y=cd)[0],
                             eqFunc=lambda a,b,cd: np.allclose(a.value(y=cd),b(y=cd)))
        #Heap of (vert, edge) pairs,
        #with invariant: all([e.isUpper() for v,e in event_list])
        event_list = [HeapWrapper(x.origin, x) for x in edgeSet.difference(lowerEdges)]
        event_list += [HeapWrapper(x.origin, x.twin) for x in lowerEdges]
        heapq.heapify(event_list)

        #populate the discovered set:
        discovered.update([x.ord for x in event_list])
        
        logging.debug("Event_list: {}".format(len(event_list)))
        #Main loop of the Intersection Algorithm
        while bool(event_list):
            logging.debug("--------------------")
            logging.debug("Event list remaining: {}".format(len(event_list)))
            logging.debug("\n".join([repr(x) for x in event_list]))
            #Get all segments that are on the current vertex
            curr_vert, curr_edge_list = pop_while_same(event_list)
            assert(isinstance(curr_vert, Vertex))
            assert(not bool(curr_edge_list) or \
                   all([isinstance(x, HalfEdge) for x in curr_edge_list]))

            sweep_y = curr_vert.toArray()[1]
            #find the nearest segment in the tree
            logging.debug("Searching tree")
            closest_node,d = status_tree.search(curr_vert.toArray(),
                                                cmpData=sweep_y,
                                                closest=True,
                                                cmpFunc=lambda a,b,cd: a.value(y=cd)[0] < b[0],
                                                eqFunc=lambda a,b,cd: np.allclose(a.value(y=cd), b))

            candidates = curr_edge_list.copy()
            upper_set = set()
            lower_set = set()
            contain_set = set()
            
            if closest_node is not None:
                logging.debug("Segment found")
                closest_segment = closest_node.value
                #Construct the starting, continuing, and ending sets

                logging.debug("Getting neighbours")
                candidate_nodes = closest_node.getNeighbours_while(partial(NEIGHBOUR_CONDITION, curr_vert))
                candidates += [x.value for x in candidate_nodes]
                
            logging.debug("Finished neighbours")
            for c in candidates:
                if c.origin == curr_vert:
                    upper_set.add(c)
                elif c.twin.origin == curr_vert:
                    lower_set.add(c)
                else:
                    contain_set.add(c)

            #----- Sets constructed

            #Report the intersections
            if sum([len(x) for x in [upper_set, lower_set, contain_set]]) > 1:
                logging.debug("Reporting intersections")
                results.append(IntersectResult(curr_vert, upper_set, contain_set, lower_set))

            #delete contain and lower sets
            logging.debug("Deleting values")
            status_tree.delete_value(*contain_set.union(lower_set), cmpData=sweep_y)

            #insert the segments with the status line a little lower
            logging.debug("Inserting values")
            candidate_lines = contain_set.union(upper_set)
            flat_lines = set([x for x in candidate_lines if x.isFlat()])
            newNodes = status_tree.insert(*candidate_lines.difference(flat_lines),
                                          cmpData=sweep_y + SWEEP_NUDGE)
            newNodes += status_tree.insert(*flat_lines,
                                           cmpData=sweep_y + SWEEP_NUDGE)

            #Calculate additional events
            logging.debug("Finding new Events")
            if closest_node is not None and (len(contain_set) + len(upper_set)) == 0:
                leftN = closest_node.getPredecessor()
                rightN = closest_node.getSuccessor()
                if leftN is not None and rightN is not None:
                    LineIntersector.findNewEvents(leftN.value,
                                                  rightN.value,
                                                  curr_vert,
                                                  event_list,
                                                  discovered)
            else:
                paired = [(a.value(y=sweep_y)[0], a) for a in newNodes]
                paired.sort(key=lambda x: x[0])

                #todo: might not be candidates, might be a sorted set access
                leftmost = paired[0][1]
                leftmostN = leftmost.getPredecessor()
                if leftmostN is not None and leftmost is not None:
                    LineIntersector.findNewEvents(leftmostN.value,
                                                  leftmost.value,
                                                  curr_vert,
                                                  event_list,
                                                  discovered)
                
                rightmost = paired[-1][1]
                rightmostN = rightmost.getSuccessor()
                
                if rightmost is not None and rightmostN is not None:
                    LineIntersector.findNewEvents(rightmost.value,
                                                  rightmostN.value,
                                                  curr_vert,
                                                  event_list,
                                                  discovered)

        return results

    @staticmethod
    def findNewEvents(a,b,vert, heap, discovered):
        assert(isinstance(vert, Vertex))
        assert(isinstance(a, HalfEdge))
        assert(isinstance(b, HalfEdge))

        logging.debug("Finding Events for: {}, {}, {}".format(a,b,vert))        
        dcel = vert.dcel
        intersection = a.intersect(b)
        matchVert = None
        if intersection is None:
            logging.debug("No intersection")
            return
        #TODO: only works on cartesian
        if intersection[1] > vert.loc[1]:
            logging.debug("Intersection too high")
            return
        if intersection[1] < vert.loc[1] or\
           (intersection[1] == vert.loc[1] and vert.loc[0] <= intersection[0]):
            logging.debug("Within bounds")
            matchVert = dcel.newVertex(intersection)
            if matchVert in discovered:
                logging.debug("Vertex already discovered")
                return
        #finally, success!:
        if matchVert is not None:
            discovered.add(matchVert)
            wrapped = HeapWrapper(matchVert, a)
            logging.debug("Adding: {}".format(wrapped))

            heapq.heappush(heap, wrapped)
            wrapped2 = HeapWrapper(matchVert, b)
            heapq.heappush(heap, wrapped2)
    








