import heapq
import logging as root_logger
from functools import partial
import IPython
import numpy as np
from collections import namedtuple
from math import inf

from ..heaputils import pop_while_same, HeapWrapper
from ..rbtree import RBTree, Directions
from ..constants import D_EPSILON
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

#------------------------------
# def Comparison functions
#------------------------------
    
def lineCmp(a, b, cd):
    """ Line comparison to be used in the status tree """
    #Is horizontal:
    aHor = a.value.isFlat()
    bHor = b.isFlat()
    #---
    logging.debug("Comparison: {} - {}".format(a.value.index, b.index))
    logging.debug("Flat:       {} - {}".format(aHor, bHor))
    y = cd['y']
    if not (aHor or bHor):
        y += cd['nudge']
    aRanges = a.value.getRanges()
    bRanges = b.getRanges()
        
    aVal = a.value(y=y)[0]
    bVal = b(y=y)[0]
    if bHor:
        bVal = min(max(cd['x'], bRanges[0,0]), bRanges[0,1])
    if aHor:
        aVal = min(max(cd['x'], aRanges[0,0]), aRanges[0,1])

    logging.debug("Values: {} - {}".format(aVal, bVal))
    
    if aVal <= bVal:
        return Directions.RIGHT
    return Directions.LEFT

def lineEq(a, b, cd):
    return a.value == b

def lineCmpVert(a, b, cd):
    result = Directions.LEFT
    if a.value.isFlat():
        aVal = cd['x']
    else:
        aVal = a.value(y=cd['y'])[0]
    logging.debug("VERT aVal{}: {}  bVal: {}".format(a.value.index,aVal, b[0]))
    
    if aVal <= b[0]:
        result = Directions.RIGHT
    return result

def lineEqVert(a,b,cd):
    return np.allclose(a.value(y=cd['y']), b)


#------------------------------
# def MAIN CLASS
#------------------------------

    
class LineIntersector:
    """ Processes a DCEL to intersect halfEdges, 
    in a self contained class
    """

    def __init__(self, dcel):
        self.dcel = dcel
        self.edgeSet = set()
        self.lowerEdges = []
        self.results = []
        self.discovered = set()
        self.sweep_y = inf
        #Tree to keep active edges in,
        #Sorted by the x's for the current y of the sweep line
        self.status_tree = RBTree(cmpFunc=lineCmp,
                                  eqFunc=lineEq)
        #Heap of (vert, edge) pairs,
        #with invariant: all([e.isUpper() for v,e in event_list])
        self.event_list = []
        
    #------------------------------
    # def MAIN CALL
    #------------------------------
    
    def __call__(self, edgeSet=None):
        self.initialise_data(edgeSet=edgeSet)
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
            self.debug_chain()

            logging.debug("Searching tree")            
            closest_node, d = self.search_tree(curr_vert)
            upper_set, contain_set, lower_set = self.determine_sets(curr_vert,
                                                                    closest_node,
                                                                    curr_edge_list.copy())
            self.report_intersections(curr_vert, upper_set, contain_set, lower_set)
            
            #todo: delete non-flat points of the non-flat event
            self.delete_values(contain_set.union(lower_set), curr_x=curr_vert.loc[0])

            self.debug_chain()
            
            #insert the segments with the status line a little lower
            candidate_lines = contain_set.union(upper_set)
            newNodes = self.insert_values(candidate_lines, curr_x=curr_vert.loc[0])

            #Calculate additional events
            self.debug_chain()
            self.handle_new_events(curr_vert, closest_node, newNodes)

        assert(self.sweep_y != inf)
        assert(not bool(self.event_list))
        return self.results


    def initialise_data(self, edgeSet=None):
        logging.debug("Starting line intersection algorithm")
        self.results = []
        #setup the set of edges to intersect
        if edgeSet is None:
            self.edgeSet = self.dcel.halfEdges.copy()
        else:
            assert(isinstance(edgeSet, set))
            #get the twins as well
            twins = [x.twin for x in edgeSet]
            edgeSet.update(twins)
            self.edgeSet = edgeSet
        assert(self.edgeSet is not None)
        self.lowerEdges = [e for e in self.edgeSet if not e.isUpper()]        
        self.event_list = [HeapWrapper(x.origin, x, desc="initial") for x in self.edgeSet.difference(self.lowerEdges)]
        self.event_list += [HeapWrapper(x.origin, x.twin, desc="initial_twin") for x in self.lowerEdges]
        heapq.heapify(self.event_list)
        self.discovered.update([x.ord for x in self.event_list])
        
        logging.debug("EdgeSet: {}, lowerEdges: {}".format(len(self.edgeSet), len(self.lowerEdges)))
        logging.debug("Event_list: {}".format(len(self.event_list)))

    def determine_sets(self, curr_vert, closest, candidates):
        assert(isinstance(curr_vert, Vertex))
        upper_set = set()
        contain_set = set()
        lower_set = set()

        if closest is not None:
            closest_segment = closest.value
            #if a line is horizontal, switch to a call(y) while, and get lines until not in horizontal bounds
            candidate_nodes = closest.getNeighbours_while(partial(NEIGHBOUR_CONDITION, curr_vert))
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

        

    def handle_new_events(self, curr_vert, closest, newNodes):
        logging.debug("Finding new Events")
        #todo: only find new events if there is a set of non-horizontal lines
        if not bool(newNodes):
            logging.debug("closest_node is not None")
            leftN = None
            rightN = None
            if closest is not None:
                leftN = closest.getPredecessor()
                rightN = closest.getSuccessor()
            logging.debug("LeftN: {}".format(leftN))
            logging.debug("RightN: {}".format(rightN))

            if leftN is not None and rightN is not None:
                self.findNewEvents(leftN.value,
                                   rightN.value,
                                   curr_vert.toArray())
        else:
            logging.debug("Closest_node is None")
            assert(bool(self.status_tree))
            paired = [(lineSortConvert(a.value,self.sweep_y+SWEEP_NUDGE, curr_vert.loc[0]), a) for a in newNodes]
            paired.sort(key=lambda x: x[0])
            logging.debug("New Nodes: {}".format([x[1].value.index for x in paired]))
            if not bool(paired):
                return
            #todo: might not be candidates, might be a sorted set access
            leftmost = paired[0][1]
            leftmostN = leftmost.getPredecessor()
            if leftmostN is not None and leftmost is not None:
                self.findNewEvents(leftmostN.value,
                                   leftmost.value,
                                   curr_vert.toArray())

                
            rightmost = paired[-1][1]
            rightmostN = rightmost.getSuccessor()
                
            if rightmost is not None and rightmostN is not None:
                self.findNewEvents(rightmost.value,
                                   rightmostN.value,
                                   curr_vert.toArray())

    def findNewEvents(self, a,b,loc):
        assert(isinstance(loc, np.ndarray))
        assert(isinstance(a, HalfEdge))
        assert(isinstance(b, HalfEdge))

        logging.debug("Finding Events for: {}, {}, {}".format(a.index,b.index,loc))        
        intersection = a.intersect(b)
        matchVert = None
        if intersection is None:
            logging.debug("No intersection")
            return
        #TODO: only works on cartesian
        if intersection[1] > loc[1]:
            logging.debug("Intersection too high")
            return
        if intersection[1] < loc[1] or\
           (intersection[1] == loc[1] and loc[0] <= intersection[0]):
            logging.debug("Within bounds")
            matchVert = self.dcel.newVertex(intersection)
            if matchVert in self.discovered:
                logging.debug("Vertex already discovered")
                return
        #finally, success!:
        if matchVert is not None:
            self.discovered.add(matchVert)
            wrapped = HeapWrapper(matchVert, a, desc="newEvent")
            wrapped2 = HeapWrapper(matchVert, b, desc="newEvent")
            logging.debug("Adding: {}".format(wrapped))
            heapq.heappush(self.event_list, wrapped)
            heapq.heappush(self.event_list, wrapped2)
    
    #------------------------------
    # def UTILITIES
    #------------------------------

    def debug_chain(self):
        chain = [x.value.index for x in self.status_tree.get_chain()]
        logging.debug("Tree Chain is: {}".format(chain))

    def report_intersections(self, v, u, c, l):
        #todo: for each horizontal crossing point separately
        if sum([len(x) for x in [u, l, c]]) > 1:
            logging.debug("Reporting intersections")
            self.results.append(IntersectResult(v, u, c, l))

    def get_next_event(self):
        #Get all segments that are on the current vertex
        curr_vert, curr_edge_list = pop_while_same(self.event_list)
        assert(not bool(curr_edge_list) or all([isinstance(x, HalfEdge) for x in curr_edge_list]))
        return curr_vert, curr_edge_list

    def delete_values(self, values, curr_x):
        logging.debug("Deleting values: {}".format([x.index for x in values]))
        assert(all([x.isUpper() for x in values]))
        self.status_tree.delete_value(*values, cmpData={'y':self.sweep_y,'x': curr_x})

    def insert_values(self, candidates, curr_x):
        logging.debug("Inserting values: {}".format([x.index for x in candidates]))
        flat_lines = set([x for x in candidates if x.isFlat()])
        #insert only non-flat lines
        toAdd = candidates.difference(flat_lines)
        assert(all([x.isUpper() for x in toAdd]))
        newNodes = self.status_tree.insert(*toAdd,
                                           cmpData={'y':(self.sweep_y + SWEEP_NUDGE), 'x': curr_x})
        newNodes += self.status_tree.insert(*flat_lines,
                                            cmpData={'y':(self.sweep_y + SWEEP_NUDGE), 'x': curr_x})

        return newNodes
        
    def update_sweep(self, curr):
        assert(isinstance(curr, Vertex))
        candidate = curr.toArray()[1]
        if candidate > self.sweep_y:
            raise Exception("Sweep line moved in wrong direction")
        self.sweep_y = candidate

    def search_tree(self, curr):
        assert(isinstance(curr, Vertex))
        closest_node,d = self.status_tree.search(curr.toArray(),
                                            cmpData=self.sweep_y,
                                            closest=True,
                                            cmpFunc=lineCmpVert,
                                            eqFunc=lineEqVert)

        if closest_node is not None:
            logging.debug("Closest Node: {}".format(closest_node.value.index))
        else:
            logging.debug("Closest Node is None")
        return closest_node, d



