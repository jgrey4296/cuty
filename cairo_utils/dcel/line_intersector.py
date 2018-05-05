import heapq
import logging as root_logger
from functools import partial

from ..heaputils import pop_while_same
from ..rbtree import RBTree
from .constants import SWEEP_NUDGE
from .Vertex import Vertex
from .HalfEdge import HalfEdge


logging = root_logger.getLogger(__name__)

class LineIntersector:
    """ Processes a DCEL to intersect halfEdges, 
    in a self contained class
    """
    @staticmethod
    def __call__(edgeSet=None, dcel=None):
        #setup the set of edges to intersect
        assert(edgeSet is not None or dcel is not None)
        if edgeSet is None:
            edgeSet = list(dcel.halfEdges.copy())
        else:
            edgeSet = list(edgeSet)
        lowerEdges = [e for e in edgeSet if not e.isUpper()]
        
        results = []
        discovered = set()
        #Tree to keep active edges in,
        #Sorted by the x's for the current y of the sweep line
        status_tree = RBTree(cmp=lambda a,b,cd: a(y=cd)[0] < b(y=cd)[0],
                             eq=lambda a,b,cd: np.allclose(a.value(y=cd),
                                                           b.toArray()))
        #Heap of (vert, edge) pairs,
        #with invariant: all([e.isUpper() for v,e in event_list])
        event_list = [(x.origin, x) for x in edgeSet.difference(lowerEdges)]
        event_list += [(x.origin, x.twin) for x in lowerEdges]
        heapq.heapify(event_list)

        #Main loop of the Intersection Algorithm
        while bool(event_list):
            #Get all segments that are on the current vertex
            curr_vert, curr_edge_list = pop_while_same(event_list)
            sweep_y = curr_vert.toArray()[1]
            #find the nearest segment in the tree
            closest_node = status_tree.search(curr_vert,
                                         cmpData=sweep_y)
            closest_segment = closest_node.value
            #Construct the starting, continuing, and ending sets
            upper_set = set(curr_edge_list)
            lower_set = set()
            contain_set = set()

            condition = lambda v, n: n.value.contains_vertex(v)
            candidates = closest_node.getNeighbours_while(partial(condition, curr_vert))
            for c in candidates:
                if c.value.origin == curr_vert:
                    upper_set.add(c.value)
                elif c.value.twin.origin == curr_vert:
                    lower_set.add(c.value)
                else:
                    contain_set.add(c.value)                
            #----- Sets constructed

            #Report the intersections
            if sum([len(x) for x in [upper_set, lower_set, contain_set]]) > 1:
                results.append((curr_vert, upper_set, contain_set, lower_set))

            #delete contain and lower sets
            status_tree.delete(*contain_set.union(lower_set))

            #insert the segments with the status line a little lower
            newNodes = status_tree.insert(*contain_set.union(upper_set),
                               cmpData=sweep_y + SWEEP_NUDGE)

            #Calculate additional events
            if (len(contain_set) + len(upper_set)) == 0:
                leftN = closest_node.getPredecessor()
                rightN = closest_node.getSuccessor()
                LineIntersector.findNewEvents(leftN.value,
                                   rightN.value,
                                   curr_vert,
                                   event_list,
                                   discovered)
            else:
                paired = [(a.value(y=sweep_y), a) for a in newNodes]
                paired.sort()
                #todo: might not be candidates, might be a sorted set access
                leftmost = paired[0][1]
                leftmostN = leftmost.getPredecessor()
                LineIntersector.findNewEvents(leftmostN.value,
                                   leftmost.value,
                                   curr_vert,
                                   event_list,
                                   discovered)
                rightmost = paired[-1][1]
                rightmostN = rightmost.getSuccessor().value
                LineIntersector.findNewEvents(rightmost.value,
                                   rightmostN.value,
                                   curr_vert,
                                   event_list,
                                   discovered)

        return results

    @staticmethod
    def findNewEvents(a,b,vert, heap, discovered):
        assert(isinstance(vert, Vertex))
        assert(instance(a, HalfEdge))
        assert(isinstance(b,HalfEdge))

        dcel = vert.dcel
        intersection = a.intersect(b)
        matchVert = None
        if intersection is None:
            return
        #TODO: only works on cartesian
        if intersection[1] > vert.loc[1]:
            return
        if intersection[1] == vert.loc[1] \
           and vert.loc[0] <= intersection[0]:
            matchVert = dcel.newVertex(intersection)
            if matchVert in discovered:
                return
        #finally, success!:
        if matchVert is not None:
            heapq.heappush(heap, (matchVert, None))
    








