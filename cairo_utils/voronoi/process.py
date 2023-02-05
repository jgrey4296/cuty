#!/usr/bin/env python3
"""

"""
##-- imports
from __future__ import annotations

import types
import abc
import datetime
import enum
import functools as ftz
import itertools as itz
import logging as logmod
import pathlib as pl
import re
import time
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Final, Generic,
                    Iterable, Iterator, Mapping, Match, MutableMapping,
                    Protocol, Sequence, Tuple, TypeAlias, TypeGuard, TypeVar,
                    cast, final, overload, runtime_checkable)
from uuid import UUID, uuid1
from weakref import ref

##-- end imports

##-- logging
logging = logmod.getLogger(__name__)
##-- end logging

class VoronoiProcess:

    def finalise_dcel(self, constrain_to_bbox=True, radius=100):
        """ Cleanup the DCEL of the voronoi diagram,
            completing faces and constraining to a bbox
        """
        if bool(self.events):
            logging.warninging("Finalising with events still to process")
        logging.debug("-------------------- Finalising DCEL")
        logging.debug(self.dcel)

        self._update_arcs(self.sweep_position.y() - 1000)
        #Not a pure DCEL operation as it requires curve intersection:
        self._complete_edges()
        self.dcel.purge()
        tempbbox = self.bbox + np.array([100, 100, -100, -100])
        #np.array([100, 100, -100, -100])
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
        if isinstance(event, SiteEvent):
            self._handle_site_event(event)
        elif isinstance(event, CircleEvent):
            if event.active:
                self._handle_circle_event(event)
            else:
                logging.debug("-------------------- Skipping deactivated circle event")
                logging.debug(event)
        else:
            raise Exception("Unrecognised Event")
        return False

    def _calculate_circle_events(self, node, left=True, right=True):
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
                self._add_circle_event(left_circle_loc, left_triple[1], left_circle[0], left=True)
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
                self._add_circle_event(right_circle_loc,
                                       right_triple[1],
                                       right_circle[0],
                                       left=False)
            else:
                logging.debug("Right points failed: {}".format(right_points))

    def _update_arcs(self, d):
        """ Trigger the update of all stored arcs with a new frontier line position """
        self.beachline.update_values(lambda v, q: v.update_d(q), d)

    #-------------------- DCEL Completion


    def _add_circle_event(self, loc, source_node, voronoi_vertex, left=True):
        if loc[1] > self.sweep_position.y():# or np.allclose(loc[1], self.sweep_position.y()):
            logging.debug("Breaking out of add circle event: Wrong side of Beachline")
            return
        event = CircleEvent(loc, source_node, voronoi_vertex, i=self.current_step, left=left)
        logging.debug("Adding: {}".format(event))
        heapq.heappush(self.events, event)
        self.circles.append(event)

    def _delete_circle_events(self, node, pre=None, post=None, event=None):
        """ Deactivate a circle event rather than deleting it.
        This means instead of removal and re-heapifying, you just skip the event
        when you come to process it """
        logging.debug("Deactivating Circle Event: {}".format(node))
        if node is not None:
            if CIRCLE_EVENTS.LEFT in node.data:
                node.data[CIRCLE_EVENTS.LEFT].deactivate()
            if CIRCLE_EVENTS.RIGHT in node.data:
                node.data[CIRCLE_EVENTS.RIGHT].deactivate()

        if pre is not None and CIRCLE_EVENTS.RIGHT in pre.data:
            pre.data[CIRCLE_EVENTS.RIGHT].deactivate()
        if post is not None and CIRCLE_EVENTS.LEFT in post.data:
            post.data[CIRCLE_EVENTS.LEFT].deactivate()



    def relax(self, amnt=0.5, faces=None):
        """ Having calculated the voronoi diagram, use the centroids of
            the faces instead of the sites, and rerun the calculation.
        Can be passed in a subset of faces
        """
        assert(not bool(self.events))

        if faces is None:
            faces = self.dcel.faces
        #Figure out any faces that are excluded
        face_indices = set([x.index for x in faces])
        other_face_sites = np.array([x.site for x in self.dcel.faces
                                     if x.index not in face_indices])
        #Get a line of origin - centroid
        lines = np.array([np.concatenate((x.site, x.getAvgCentroid())) for x in faces])
        #Move along that line toward the centroid
        new_sites = np.array([utils.math.sample_along_lines(*x, amnt)[0] for x in lines])
        #Combine with excluded faces
        if bool(other_face_sites) and bool(new_sites):
            total_sites = np.row_stack((new_sites, other_face_sites))
        elif bool(new_sites):
            total_sites = new_sites
        else:
            total_sites = other_face_sites
        assert(len(self.dcel.faces) == len(total_sites))
        #Setup the datastructures with the new sites
        self.reset()
        self.init_graph(data=new_sites, rerun=True)
        self.calculate_to_completion()


class _VoronoiHandlers:

    def _handle_site_event(self, event):
        """
        provided with a site event, add it to the beachline in the appropriate place
        then update/remove any circle events that trios of arcs generate
        """
        assert(isinstance(event, SiteEvent))
        logging.debug("Handling Site Event: {}".format(event))
        #The new parabola made from the site
        new_arc = Parabola(*event.loc, self.sweep_position.y())
        #get the x position of the event
        x_pos = new_arc.fx

        #if beachline is empty: add and return
        if not bool(self.beachline):
            new_node = self.beachline.insert(new_arc)[0]
            new_node.data['face'] = event.face
            return

        #Otherwise, slot the arc between existing nodes
        closest_node, direction = self._get_closest_arc_node(x_pos)
        assert(closest_node is not None)
        #remove the obsolete circle event
        self._delete_circle_events(closest_node)
        new_node, duplicate_node = self._split_beachline(direction,
                                                         closest_node,
                                                         new_arc,
                                                         event.face)

        #Create an edge between the two nodes, without origin points yet
        logging.debug("Adding edge on side: {}".format(direction))
        node_face = closest_node.data['face']
        if direction is Directions.LEFT:
            the_face = event.face
            twin_face = node_face
            node_pair = (new_node, closest_node)
        else:
            the_face = node_face
            twin_face = event.face
            node_pair = (closest_node, new_node)

        new_edge = self.dcel.new_edge(None, None, face=the_face, twin_face=twin_face,
                                      edata=BASE_VORONOI_EDGE_DATA, vdata=BASE_VORONOI_VERT_DATA)
        self._store_edge(new_edge, *node_pair)
        self._cleanup_edges(direction, new_edge, new_node, closest_node, duplicate_node)

        #create circle events:
        self._calculate_circle_events(new_node)

    def _handle_circle_event(self, event):
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
        new_vertex = self.dcel.new_vertex(event.vertex, data=BASE_VORONOI_VERT_DATA)

        #attach the vertex as a defined point in the half edges for the three faces,
        #these faces are pre<->node and node<->succ

        e1 = self._get_edge(pre, node)
        e2 = self._get_edge(node, suc)

        #create two half-edge records for the new breakpoint of the beachline
        logging.debug("Creating a new half-edge {}-{}".format(pre, suc))
        new_edge = self.dcel.new_edge(new_vertex, None,
                                      face=pre.data['face'],
                                      twin_face=suc.data['face'],
                                      edata=BASE_VORONOI_EDGE_DATA,
                                      vdata=BASE_VORONOI_VERT_DATA)

        if e1:
            #predecessor face
            logging.debug("Adding vertex to {}-{}".format(pre, node))
            assert(e1.face == pre.data['face'])
            assert(e1.twin.face == node.data['face'])
            e1.addVertex(new_vertex)
            e1.addPrev(new_edge, force=True)
        else:
            logging.debug("No r-edge found for {}-{}".format(pre, node))

        if e2:
            #successor face
            logging.debug("Adding vertex to {}-{}".format(node, suc))
            assert(e2.twin.face == suc.data['face'])
            assert(e2.face == node.data['face'])
            e2.addVertex(new_vertex)
            e2.twin.addNext(new_edge.twin, force=True)
        else:
            logging.debug("No r-edge found for {}-{}".format(node, suc))

        #store the new edge, but only for the open breakpoint
        #the breakpoint being the predecessor and successor, now partners following
        #removal of the middle node above in this function
        self._store_edge(new_edge, pre, suc)

        #delete the node, no longer needed as the arc has reduced to 0
        logging.debug("Pre-Deletion: {}".format(self.beachline.get_chain()))
        self.beachline.delete(node)
        logging.debug("Post-Deletion: {}".format(self.beachline.get_chain()))
        #recheck for new circle events
        if pre:
            self._calculate_circle_events(pre, left=False, right=True)
        if suc:
            self._calculate_circle_events(suc, left=True, right=False)

class _VoronoiUtils:

    def _cleanup_edges(self, direction, edge, new_node, node, duplicate_node):
        """ if there was an edge of closest_arc -> closest_arc.successor: update it
        because closest_arc is not adjacent to successor any more, duplicate_node is """
        if direction is Directions.LEFT:
            logging.debug("Cleaning up left")
            dup_node_sibling = duplicate_node.getPredecessor()
            if dup_node_sibling is not None:
                e1 = self._get_edge(dup_node_sibling, node)
                if e1 is not None:
                    self._remove_edge(dup_node_sibling, node)
                    self._store_edge(e1, dup_node_sibling, duplicate_node)
        else:
            logging.debug("Cleaning up right")
            dup_node_sibling = duplicate_node.getSuccessor()
            if dup_node_sibling is not None:
                e1 = self._get_edge(node, dup_node_sibling)
                if e1 is not None:
                    self._remove_edge(node, dup_node_sibling)
                    self._store_edge(e1, duplicate_node, dup_node_sibling)

        if direction is Directions.LEFT:
            self._store_edge(edge, new_node, node)
            self._store_edge(edge.twin, duplicate_node, new_node)
        else:
            self._store_edge(edge, node, new_node)
            self._store_edge(edge.twin, new_node, duplicate_node)

    def _get_closest_arc_node(self, x_pos):
        #search for the breakpoint interval of the beachline
        closest_arc_node, direction = self.beachline.search(x_pos, closest=True)
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
        triple_string = "-".join([repr(x) for x in [node, new_node, duplicate_node]])
        logging.debug("Split {} into {}".format(repr(node), triple_string))
        return new_node, duplicate_node

    #-------------------- Fortune Methods

    def _complete_edges(self):
        """ get any infinite edges, and complete their intersections """
        logging.debug("\n---------- Infinite Edges Completion")
        i = 0

        #get only the halfedges that are originless, rather than full edges that are infinite
        i_pairs = [x for x in self.halfedges.items() if x[1].isInfinite()]
        logging.debug("Origin-less half edges num: {}".format(len(i_pairs)))

        #----
        #i_pairs = [((breakpoint nodes), edge)]
        for (bw, c) in i_pairs:
            i += 1
            #a and b are nodes
            a = bw.bp1
            b = bw.bp2
            debug_string = "{} Infinite Edge resolution: {}-{}, infinite? {}"
            logging.debug(debug_string.format(i, a, b, c.isInfinite()))
            if c.origin is None and c.twin.origin is None:
                logging.debug("Found an undefined edge, cleaning up")
                c.markForCleanup()
                continue
            if not c.isInfinite():
                continue
            #raise Exception("An Edge is not infinite")
            #intersect the breakpoints to find the vertex point
            intersection = a.value.intersect(b.value)
            if intersection is None or not bool(intersection):
                raise Exception("No intersections detected when completing an infinite edge")
            elif len(intersection) == 2:
                verts = [x for x in c.getVertices() if x is not None]
                assert(len(verts) == 1)
                lines = []
                lines += bound_line_in_bbox(np.array([verts[0].toArray(), intersection[0]]),
                                            self.bbox)
                lines += bound_line_in_bbox(np.array([verts[0].toArray(), intersection[1]]),
                                            self.bbox)
                distances = np.array([get_distance_raw(x[:2], x[2:]) for x in lines])
                min_line = np.argmin(distances)
                new_vertex = self.dcel.new_vertex(lines[min_line][1], data=BASE_VORONOI_VERT_DATA)
                c.addVertex(new_vertex)

            if c.isInfinite():
                logging.debug("Edge is still infinite, marking for cleanup")
                c.markForCleanup()

    #-------------------- Beachline Edge Interaction

    def _store_edge(self, edge, bp1, bp2):
        """ Store an incomplete edge by the 2 pairs of nodes that define the breakpoints """
        assert(isinstance(edge, HalfEdge))
        assert(isinstance(bp1, rbtree.Node))
        assert(isinstance(bp2, rbtree.Node))
        if self._has_edge(bp1, bp2) and self._get_edge(bp1, bp2) != edge:
            raise Exception("Overrighting edge breakpoint: {}, {}".format(bp1, bp2))
        logging.debug("Storing Edge: ({}, {}): {}".format(bp1, bp2, edge))
        self.halfedges[BreakWrapper(bp1, bp2)] = edge

    def _has_edge(self, bp1, bp2):
        assert(bp1 is None or isinstance(bp1, rbtree.Node))
        assert(bp2 is None or isinstance(bp2, rbtree.Node))
        return BreakWrapper(bp1, bp2) in self.halfedges

    def _get_edge(self, bp1, bp2):
        assert(bp1 is None or isinstance(bp1, rbtree.Node))
        assert(bp2 is None or isinstance(bp2, rbtree.Node))
        if self._has_edge(bp1, bp2):
            return self.halfedges[BreakWrapper(bp1, bp2)]
        else:
            return None

    def _remove_edge(self, bp1, bp2):
        assert(isinstance(bp1, rbtree.Node))
        assert(isinstance(bp2, rbtree.Node))
        if not self._has_edge(bp1, bp2):
            raise Exception("trying to remove a non-existing edge")
        logging.debug("Removing Edge: ({}, {}) : {}".format(bp1,
                                                            bp2,
                                                            self.halfedges[BreakWrapper(bp1, bp2)]))
        del self.halfedges[BreakWrapper(bp1, bp2)]

    #-------------------- Circle Event Interaction
