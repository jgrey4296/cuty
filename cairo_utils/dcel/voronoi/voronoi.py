""" Voronoi.py : Contains the Voronoi Class, which calculates a graphics independent DCEL
    of a Voronoi Diagram.
"""
import numpy as np
import numpy.random as random
import heapq
import pickle
import logging as root_logger
import sys
import IPython
from os.path import isfile
from string import ascii_uppercase
from math import pi, sin, cos

import cairo_utils as utils
from cairo_utils import Parabola
from cairo_utils import rbtree
from cairo_utils.rbtree.ComparisonFunctions import arc_comparison, Directions, arc_equality

from cairo_utils.dcel import DCEL, HalfEdge, Face
from cairo_utils.math import get_distance_raw, bound_line_in_bbox, isClockwise, bbox_centre

from .Events import SiteEvent, CircleEvent, VEvent, CIRCLE_EVENTS, arc_cleanup
from .voronoi_drawing import Voronoi_Debug

logging = root_logger.getLogger(__name__)

#Constants and defaults
image_dir = "imgs"
SAVENAME = "graph_data.pkl"
BBOX = np.array([0,0,1,1]) #the bbox of the final image
EPSILON = sys.float_info.epsilon
MAX_STEPS = 100000
CARTESIAN = True

base_voronoi_vert_data = { "VORONOI_VERTEX" : True}
base_voronoi_edge_data = {"VORONOI_EDGE" : True}
base_voronoi_face_data = {"VORONOI_FAE" : True}

class Voronoi:
    """ Creates a random selection of points, and step by step constructs
        a voronoi diagram
    """
    def __init__(self, num_of_nodes=10, bbox=BBOX, save_name=SAVENAME,
                 debug_draw=False, n=10, max_steps=MAX_STEPS, dcel=None):
        assert(isinstance(sizeTuple, tuple))
        assert(isinstance(bbox, np.ndarray))
        assert(bbox.shape == (4,))
        self.current_step = 0
        self.nodeSize = num_of_nodes
        self.max_steps = max_steps
        #The output voronoi diagram as a DCEL
        if dcel is None:
            self.__protect_dcel = False
            self.dcel = DCEL(bbox=bbox)
        else:
            self.__protect_dcel = True
            self.dcel = dcel
            if self.bbox != self.dcel.bbox:
                self.bbox = self.bbox
        #Min Heap of site/circle events
        self.events = []
        #backup of the original sites
        self.sites = []
        #backup of all circle events
        self.circles = []
        #storage of breakpoint tuples -> halfedge
        self.halfEdges = {}
        #The bbox of the diagram
        self.bbox = bbox
        VEvent.offset = self.bbox[3] - self.bbox[1]
        
        #The Beach Line Data Structure
        self.beachline = None
        #The sweep line position
        self.sweep_position = None
            

        #File name to pickle data to:
        self.save_file_name = save_name
        
        self.debug_draw = debug_draw
        if self.debug_draw:
            self.debug = Voronoi_Debug(n, image_dir, self)
        
    #--------------------
    # PUBLIC METHODS
    #--------------------
    def reset(self):
        """ Reset the internal data structures """
        if not self.__protect_dcel:
            self.dcel = DCEL(bbox=self.bbox)
        self.events = []
        self.circles = []
        self.halfEdges = {}
        self.sweep_position = None
        self.beachline = rbtree.RBTree(cmpFunc=arc_comparison,
                                       eqFunc=arc_equality)

        self.current_step = 0

        
    def initGraph(self,data=None,rerun=False):
        """ Create a graph of initial random sites """
        logging.debug("Initialising graph")
        self.reset()

        values = data
        if values is None and not rerun:
            values = self.load_graph()

        assert(values is None or isinstance(values, np.ndarray))
        #create a (n,2) array of coordinates for the sites, if no data has been loaded
        if values is None:
            logging.debug("Generating values")
            for n in range(self.nodeSize):
                rndAmnt = random.random((1,2))
                #scale the new site
                scaler = self.bbox.reshape((2,2)).transpose()
                newSite = scaler[:,0] + (rndAmnt * (scaler[:,1] - scaler[:,0]))
                if values is None:
                    values = newSite
                else:
                    values = np.row_stack((values,newSite))

        #setup the initial site events:
        usedCoords = []
        for site in values:
            #Avoid duplications:
            if (site[0],site[1]) in usedCoords:
                logging.warn("Skipping Duplicate: {}".format(site))
                continue
            #Create an empty face for the site
            futureFace = self.dcel.newFace(site, data=base_voronoi_face_data)
            event = SiteEvent(site,face=futureFace)
            heapq.heappush(self.events,event)
            self.sites.append(event)
            usedCoords.append((site[0],site[1]))

        #Save the nodes
        if not rerun:
            self.save_graph(values)
        return values

    def relax(self, amnt=0.5, faces=None):
        """ Having calculated the voronoi diagram, use the centroids of 
            the faces instead of the sites, and rerun the calculation.
        Can be passed in a subset of faces
        """
        assert(not bool(self.events))

        if faces is None:
            faces = self.dcel.faces
        #Figure out any faces that are excluded
        faceIndices = set([x.index for x in faces])
        otherFaceSites = np.array([x.site for x in self.dcel.faces if x.index not in faceIndices])
        #Get a line of origin - centroid
        lines = np.array([np.concatenate((x.site, x.getAvgCentroid())) for x in faces])
        #Move along that line toward the centroid
        newSites = np.array([utils.math.sampleAlongLine(*x, amnt)[0] for x in lines])
        #Combine with excluded faces
        if len(otherFaceSites) > 0 and len(newSites) > 0:
            totalSites = np.row_stack((newSites, otherFaceSites))
        elif len(newSites) > 0:
            totalSites = newSites
        else:
            totalSites = otherFaceSites
        assert(len(self.dcel.faces) == len(totalSites))
        #Setup the datastructures with the new sites
        self.reset()
        self.initGraph(data=newSites,rerun=True)
        self.calculate_to_completion()
        
    def calculate_to_completion(self):
        """ Calculate the entire voronoi for all points """
        finished = False
        #Max Steps for a guaranteed exit
        while not finished and self.current_step < self.max_steps:
            logging.debug("----------------------------------------")
            logging.debug("Calculating step: {}".format(self.current_step))
            finished = self._calculate()
            if self.debug_draw:
                self.debug.draw_intermediate_states(self.current_step, dcel=True, text=True)
            self.current_step += 1

    def finalise_DCEL(self, constrain_to_bbox=True, radius=100):
        """ Cleanup the DCEL of the voronoi diagram, 
            completing faces and constraining to a bbox 
        """
        if bool(self.events):
            logging.warning("Finalising with events still to process")
        logging.debug("-------------------- Finalising DCEL")
        logging.debug(self.dcel)
        
        self._update_arcs(self.sweep_position.y() - 1000)
        #Not a pure DCEL operation as it requires curve intersection:
        self._complete_edges()
        self.dcel.purge()
        tempbbox = self.bbox + np.array([100,100,-100,-100])
        #np.array([100,100,-100,-100])
        if constrain_to_bbox:
            #modify or mark edges outside bbox
            self.dcel.constrain_to_bbox(tempbbox, force=True)
        else:
            centre = bbox_centre(self.bbox)
            self.dcel.constrain_to_circle(centre, radius)
        self.dcel.purge()
        logging.debug("---------- Constrained to bbox")
        #ensure CCW ordering
        for f in self.dcel.faces:
            f.fixup(tempbbox)
        #cleanup faces
        logging.debug("---------- Fixed up faces")
        self.dcel.purge()
        logging.debug("---------- Purged 3")
        logging.debug(self.dcel)
        self.dcel.verify_all()
        return self.dcel

    def save_graph(self,values):
        with open(self.save_file_name,'wb') as f:
            pickle.dump(values,f)
        
    def load_graph(self):
        if isfile(self.save_file_name):
            with open(self.save_file_name,'rb') as f:
                return pickle.load(f)



    #--------------------
    # PRIVATE METHODS
    #--------------------
    def _calculate(self):
        """ Calculate the next step of the voronoi diagram,
            Return True on completion, False otherwise
        """
        if not bool(self.events): #finished calculating, early exit
            return True
        ##Get the next event
        event = heapq.heappop(self.events)
        #update the sweep position
        self.sweep_position = event
        logging.debug("Sweep position: {}".format(self.sweep_position.loc))
        #update the arcs:
        self._update_arcs(self.sweep_position.y())
        #handle the event:
        if isinstance(event,SiteEvent):
            self._handleSiteEvent(event)
        elif isinstance(event,CircleEvent):
            if event.active:
                self._handleCircleEvent(event)
            else:
                logging.debug("-------------------- Skipping deactivated circle event")
                logging.debug(event)
        else:
            raise Exception("Unrecognised Event")
        return False 
        
    
    #----------
    # MAIN VORONOI CALLS
    #----------
    def _handleSiteEvent(self,event):
        """
        provided with a site event, add it to the beachline in the appropriate place
        then update/remove any circle events that trios of arcs generate
        """
        assert(isinstance(event, SiteEvent))
        logging.debug("Handling Site Event: {}".format(event))
        #The new parabola made from the site
        new_arc = Parabola(*event.loc,self.sweep_position.y())
        #get the x position of the event
        xPos = new_arc.fx

        #if beachline is empty: add and return
        if not bool(self.beachline):
            newNode = self.beachline.insert(new_arc)[0]
            newNode.data['face'] = event.face
            return

        #Otherwise, slot the arc between existing nodes
        closest_node, direction = self._get_closest_arc_node(xPos)
        assert(closest_node is not None)
        #remove the obsolete circle event
        self._delete_circle_events(closest_node)
        new_node, duplicate_node  = self._split_beachline(direction,
                                                         closest_node,
                                                         new_arc,
                                                         event.face)
        
        #Create an edge between the two nodes, without origin points yet
        logging.debug("Adding edge on side: {}".format(direction))
        node_face = closest_node.data['face']
        if direction is Directions.LEFT:
            theFace = event.face
            twinFace = node_face
            nodePair = (new_node, closest_node)
        else:
            theFace = node_face
            twinFace = event.face
            nodePair = (closest_node, new_node)
        
        newEdge = self.dcel.newEdge(None, None, face=theFace, twinFace=twinFace,
                                    edata=base_voronoi_edge_data, vdata=base_voronoi_vert_data)
        self._storeEdge(newEdge, *nodePair)
        self._cleanup_edges(direction, newEdge, new_node, closest_node, duplicate_node)

        #create circle events:
        self._calculate_circle_events(new_node)

    def _handleCircleEvent(self,event):
        """
        provided a circle event, add a new vertex to the dcel, 
        then update the beachline to connect the two sides of the arc that has disappeared
        """
        assert(isinstance(event, CircleEvent))
        logging.debug("Handling Circle Event: {}".format(event))
        #remove disappearing arc from tree
        #and update breakpoints, remove false alarm circle events
        node = event.source
        pre = node.getPredecessor()
        suc = node.getSuccessor()
        assert('face' in pre.data)
        assert('face' in suc.data)

        self._delete_circle_events(node, pre, suc, event)
        
        #add the centre of the circle causing the event as a vertex record
        logging.debug("Creating Vertex")
        newVertex = self.dcel.newVertex(event.vertex, data=base_voronoi_vert_data)

        #attach the vertex as a defined point in the half edges for the three faces,
        #these faces are pre<->node and node<->succ

        e1 = self._getEdge(pre,node)
        e2 = self._getEdge(node,suc)

        #create two half-edge records for the new breakpoint of the beachline
        logging.debug("Creating a new half-edge {}-{}".format(pre,suc))
        newEdge = self.dcel.newEdge(newVertex, None,
                                    face=pre.data['face'],
                                    twinFace=suc.data['face'],
                                    edata=base_voronoi_edge_data,
                                    vdata=base_voronoi_vert_data)

        if e1:
            #predecessor face
            logging.debug("Adding vertex to {}-{}".format(pre,node))
            assert(e1.face == pre.data['face'])
            assert(e1.twin.face == node.data['face'])
            e1.addVertex(newVertex)
            e1.addPrev(newEdge, force=True)
        else:
            logging.debug("No r-edge found for {}-{}".format(pre,node))
            IPython.embed(simple_prompt=True)
            
        if e2:
            #successor face
            logging.debug("Adding vertex to {}-{}".format(node,suc))
            assert(e2.twin.face == suc.data['face'])
            assert(e2.face == node.data['face'])
            e2.addVertex(newVertex)
            e2.twin.addNext(newEdge.twin, force=True)
        else:
            logging.debug("No r-edge found for {}-{}".format(node,suc))
            IPython.embed(simple_prompt=True)
        
        #store the new edge, but only for the open breakpoint
        #the breakpoint being the predecessor and successor, now partners following
        #removal of the middle node above in this function  
        self._storeEdge(newEdge,pre,suc)

        #delete the node, no longer needed as the arc has reduced to 0
        logging.debug("Pre-Deletion: {}".format(self.beachline.get_chain()))
        self.beachline.delete(node)
        logging.debug("Post-Deletion: {}".format(self.beachline.get_chain()))
        #recheck for new circle events
        if pre:
            self._calculate_circle_events(pre,left=False, right=True)
        if suc:
            self._calculate_circle_events(suc,left=True, right=False)
        

    #----------
    # UTILITIES
    #----------        
    def _cleanup_edges(self, direction, edge, new_node, node, duplicate_node):
        """ if there was an edge of closest_arc -> closest_arc.successor: update it
        because closest_arc is not adjacent to successor any more, duplicate_node is """
        if direction is Directions.LEFT:
            logging.debug("Cleaning up left")
            dup_node_sibling = duplicate_node.getPredecessor()
            if dup_node_sibling is not None:
                e1 = self._getEdge(dup_node_sibling, node)
                if e1 is not None:
                    self._removeEdge(dup_node_sibling, node)
                    self._storeEdge(e1,dup_node_sibling, duplicate_node)
        else:
            logging.debug("Cleaning up right")
            dup_node_sibling = duplicate_node.getSuccessor()
            if dup_node_sibling is not None:
                e1 = self._getEdge(node, dup_node_sibling)
                if e1 is not None:
                    self._removeEdge(node, dup_node_sibling)
                    self._storeEdge(e1, duplicate_node, dup_node_sibling)
                
        if direction is Directions.LEFT:
            self._storeEdge(edge, new_node, node)
            self._storeEdge(edge.twin, duplicate_node, new_node)
        else:
            self._storeEdge(edge, node, new_node)
            self._storeEdge(edge.twin, new_node, duplicate_node)
        
    def _get_closest_arc_node(self, xPos):
        #search for the breakpoint interval of the beachline
        closest_arc_node, direction = self.beachline.search(xPos, closest=True)
        if closest_arc_node is not None:
            logging.debug("Closest Arc Triple: {} *{}* {}".format(closest_arc_node.getPredecessor(),
                                                                  closest_arc_node,
                                                                  closest_arc_node.getSuccessor()))
            logging.debug("Direction: {}".format(direction))
        return (closest_arc_node, direction)


    def _split_beachline(self, direction, node, arc, event_face):
        #If site is directly below the arc, or on the right of the arc, add it as a successor
        if direction is Directions.RIGHT:
            new_node = self.beachline.insert_successor(node, arc)
            duplicate_node = self.beachline.insert_successor(new_node, node.value)
        else:
            #otherwise add it as a predecessor
            new_node = self.beachline.insert_predecessor(node, arc)
            duplicate_node = self.beachline.insert_predecessor(new_node, node.value)
        assert(isinstance(new_node, rbtree.Node))
        
        #add in the faces as a data point for the new node and duplicate
        new_node.data['face'] = event_face
        duplicate_node.data['face'] = node.data['face']

        #Debug the new triple: [ A, B, A]
        tripleString = "-".join([repr(x) for x in [node,new_node,duplicate_node]])
        logging.debug("Split {} into {}".format(repr(node),tripleString))
        return new_node, duplicate_node


    #-------------------- Fortune Methods
    def _calculate_circle_events(self,node,left=True,right=True):
        """
        Given an arc node, get the arcs either side, and determine if/when it will disappear
        """
        logging.debug("Calculating circle events for: {}".format(node))
        #Generate a circle event for left side, and right side
        left_triple = self.beachline.get_predecessor_triple(node)
        right_triple = self.beachline.get_successor_triple(node)
        #Calculate chords and determine circle event point:
        #add circle event to events and the relevant leaf
        if left_triple:
            logging.debug("Calc Left Triple: {}".format("-".join([str(x) for x in left_triple])))
            
        if left and left_triple and left_triple[0].value != left_triple[2].value:
            left_points = np.array([x.value.get_focus() for x in left_triple])
            left_circle = utils.math.get_circle_3p(*left_points)

            #possibly use ccw for this, with glpoc from below
            if left_circle is not None and isClockwise(*left_points):
                left_circle_loc = utils.math.get_lowest_point_on_circle(*left_circle)
                #check the l_t_p/s arent in this circle
                #note: swapped this to add on the right ftm
                self._add_circle_event(left_circle_loc,left_triple[1],left_circle[0],left=True)
            else:
                logging.debug("Left points failed: {}".format(left_points))

        if right_triple:
            logging.debug("Calc Right Triple: {}".format("-".join([str(x) for x in right_triple])))
            
        if right and right_triple and right_triple[0].value != right_triple[2].value:
            right_points = np.array([x.value.get_focus() for x in right_triple])
            right_circle = utils.math.get_circle_3p(*right_points)
            if right_circle is not None and isClockwise(*right_points):
                right_circle_loc = utils.math.get_lowest_point_on_circle(*right_circle)
                #note: swapped this to add on the left ftm
                self._add_circle_event(right_circle_loc,right_triple[1],right_circle[0], left=False)
            else:
                logging.debug("Right points failed: {}".format(right_points))

    def _update_arcs(self,d):
        """ Trigger the update of all stored arcs with a new frontier line position """
        self.beachline.update_values(lambda v,q: v.update_d(q) ,d)

    #-------------------- DCEL Completion
    def _complete_edges(self):
        """ get any infinite edges, and complete their intersections """
        logging.debug("\n---------- Infinite Edges Completion")
        i = 0
        
        #get only the halfedges that are originless, rather than full edges that are infinite
        i_pairs = [x for x in self.halfEdges.items() if x[1].isInfinite()]
        logging.debug("Origin-less half edges num: {}".format(len(i_pairs)))
        
        #----
        #i_pairs = [((breakpoint nodes),edge)]
        for (bw,c) in i_pairs:
            i += 1
            #a and b are nodes
            a = bw.bp1
            b = bw.bp2
            logging.debug("{} Infinite Edge resolution: {}-{}, infinite? {}".format(i,a,b,c.isInfinite()))
            if c.origin is None and c.twin.origin is None:
                logging.debug("Found an undefined edge, cleaning up")
                c.markForCleanup()
                continue
            if not c.isInfinite():
                continue
            #raise Exception("An Edge is not infinite")
            #intersect the breakpoints to find the vertex point
            intersection = a.value.intersect(b.value)
            if intersection is None or len(intersection) < 1:
                raise Exception("No intersections detected when completing an infinite edge")
            elif len(intersection) == 2:
                verts = [x for x in c.getVertices() if x is not None]
                assert(len(verts) == 1)
                lines = []
                lines += bound_line_in_bbox(np.array([verts[0].toArray(), intersection[0]]),
                                        self.bbox)
                lines += bound_line_in_bbox(np.array([verts[0].toArray(), intersection[1]]),
                                        self.bbox)
                distances = np.array([get_distance_raw(*x) for x in lines])
                minLine = np.argmin(distances)
                newVertex = self.dcel.newVertex(lines[minLine][1], data=base_voronoi_vert_data)
                c.addVertex(newVertex)
            
            if c.isInfinite():
                logging.debug("Edge is still infinite, marking for cleanup")
                c.markForCleanup()

            
    #-------------------- Beachline Edge Interaction
    def _storeEdge(self,edge,bp1,bp2):
        """ Store an incomplete edge by the 2 pairs of nodes that define the breakpoints """
        assert(isinstance(edge, HalfEdge))
        assert(isinstance(bp1, rbtree.Node))
        assert(isinstance(bp2, rbtree.Node))
        if self._hasEdge(bp1,bp2) and self._getEdge(bp1,bp2) != edge:
            raise Exception("Overrighting edge breakpoint: {}, {}".format(bp1, bp2))
        logging.debug("Storing Edge: ({},{}): {}".format(bp1, bp2, edge))
        self.halfEdges[BreakWrapper(bp1,bp2)] = edge
        
    def _hasEdge(self,bp1,bp2):
        assert(bp1 is None or isinstance(bp1, rbtree.Node))
        assert(bp2 is None or isinstance(bp2, rbtree.Node))
        return BreakWrapper(bp1,bp2) in self.halfEdges

    def _getEdge(self,bp1,bp2):
        assert(bp1 is None or isinstance(bp1, rbtree.Node) )
        assert(bp2 is None or isinstance(bp2, rbtree.Node))
        if self._hasEdge(bp1,bp2):
            return self.halfEdges[BreakWrapper(bp1,bp2)]
        else:
            return None

    def _removeEdge(self,bp1,bp2):
        assert(isinstance(bp1, rbtree.Node))
        assert(isinstance(bp2, rbtree.Node))
        if not self._hasEdge(bp1,bp2):
            raise Exception("trying to remove a non-existing edge")
        logging.debug("Removing Edge: ({},{}) : {}".format(bp1,bp2, self.halfEdges[BreakWrapper(bp1,bp2)]))
        del self.halfEdges[BreakWrapper(bp1,bp2)]

    #-------------------- Circle Event Interaction
    def _add_circle_event(self,loc,sourceNode,voronoiVertex,left=True):
        if loc[1] > self.sweep_position.y():# or np.allclose(loc[1],self.sweep_position.y()):
            logging.debug("Breaking out of add circle event: Wrong side of Beachline")
            return
        event = CircleEvent(loc,sourceNode,voronoiVertex,i=self.current_step, left=left)
        logging.debug("Adding: {}".format(event))
        heapq.heappush(self.events,event)
        self.circles.append(event)

    def _delete_circle_events(self,node, pre=None, post=None, event=None):
        """ Deactivate a circle event rather than deleting it.
        This means instead of removal and re-heapifying, you just skip the event
        when you come to process it """
        logging.debug("Deactivating Circle Event: {}".format(node))
        if node is not None:
            if CIRCLE_EVENTS.LEFT in node.data:
                node.data[CIRCLE_EVENTS.LEFT].deactivate()
            if CIRCLE_EVENTS.RIGHT in node.data:
                node.data[CIRCLE_EVENTS.RIGHT].deactivate()

        if pre != None and CIRCLE_EVENTS.RIGHT in pre.data:
            pre.data[CIRCLE_EVENTS.RIGHT].deactivate()
        if post != None and CIRCLE_EVENTS.LEFT in post.data:
            post.data[CIRCLE_EVENTS.LEFT].deactivate()
        
class BreakWrapper:
    """ A Simple Breakpoint Wrapper """

    def __init__(self, bp1, bp2):
        self.bp1 = bp1
        self.bp2 = bp2
        
    def __eq__(self, other):
        assert(isinstance(other, BreakWrapper))
        if self.bp1 == other.bp1 and self.bp2 == other.bp2:
            return True
        if self.bp1.value == other.bp1.value and self.bp2.value == other.bp2.value:
            return True
        return False

    def __hash__(self):
        return hash((self.bp1, self.bp2))
