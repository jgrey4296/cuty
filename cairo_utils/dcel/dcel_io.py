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

class DCELio:

    def copy(self):
        """ Completely duplicate the dcel """
        new_dcel = DCEL(self.bbox)
        new_dcel.import_data(self.export_data())
        return new_dcel

    def export_data(self):
        """ Export a simple format to define vertices, halfedges, faces,
        uses identifiers instead of objects, allows reconstruction """
        data = {
            'vertices'   : [x._export() for x in self.vertices],
            'half_edges' : [x._export() for x in self.half_edges],
            'faces'      : [x._export() for x in self.faces],
            'bbox'       : self.bbox
        }
        return data

    def import_data(self, data):
        """ Import the data format of identifier links to actual links,
        of export output from a dcel """
        #pylint: disable=too-many-locals
        #pylint: disable=too-many-statements
        assert(all([x in data for x in ['vertices', 'half_edges', 'faces', 'bbox']]))
        self.bbox = data['bbox']

        #dictionarys used to store {newIndex : (newIndex, newObject, oldData)}
        local_vertices = {}
        local_edges = {}
        local_faces = {}
        output_mapping = {}

        #-----
        # Reconstruct Verts, Edges, Faces:
        #-----
        #construct vertices by index
        logging.info("Re-Creating Vertices: {}".format(len(data['vertices'])))
        for v_data in data['vertices']:
            combined_data = {}
            combined_data.update({VertE.__members__[a] : b for a, b in v_data['enum_data'].items()})
            combined_data.update(v_data['non_enum_data'])

            new_vert = Vertex(np.array([v_data['x'], v_data['y']]),
                              index=v_data['i'], data=combined_data,
                              dcel=self, active=v_data['active'])
            logging.debug("Re-created vertex: {}".format(new_vert.index))
            local_vertices[new_vert.index] = DataPair(new_vert.index, new_vert, v_data)

        #edges by index
        logging.info("Re-Creating HalfEdges: {}".format(len(data['half_edges'])))
        for e_data in data['half_edges']:
            combined_data = {}
            combined_data.update({EdgeE.__members__[a] : b for a, b in e_data['enum_data'].items()})
            combined_data.update(e_data['non_enum_data'])
            new_edge = HalfEdge(index=e_data['i'], data=combined_data, dcel=self)
            logging.debug("Re-created Edge: {}".format(new_edge.index))
            local_edges[new_edge.index] = DataPair(new_edge.index, new_edge, e_data)

        #faces by index
        logging.info("Re-Creating Faces: {}".format(len(data['faces'])))
        for f_data in data['faces']:
            combined_data = {}
            combined_data.update({FaceE.__members__[a] : b for a, b in f_data['enum_data'].items()})
            combined_data.update(f_data['non_enum_data'])
            new_face = Face(site=np.array([f_data['sitex'], f_data['sitey']]), index=f_data['i'],
                            data=combined_data, dcel=self)
            logging.debug("Re-created face: {}".format(new_face.index))
            local_faces[new_face.index] = DataPair(new_face.index, new_face, f_data)

        #-----
        # def Upon reconstruction, reattach ids to the same objects
        #-----
        #this only update standard connections, not user connections
        #TODO: PASS OUT A MAPPING OF OLD IDS TO NEW FOR USER UPDATES
        try:
            #connect vertices to their edges
            for vertex in local_vertices.values():
                vertex.obj.half_edges.update( \
                    [local_edges[x].obj for x in vertex.data['half_edges']])
        except Exception:
            logging.warning("Import Error for vertex")

        try:
            #connect edges to their vertices, and neighbours, and face
            for edge in local_edges.values():
                if edge.data['origin'] is not None:
                    edge.obj.origin = local_vertices[edge.data['origin']].obj
                if edge.data['twin'] is not None:
                    edge.obj.twin = local_edges[edge.data['twin']].obj
                if edge.data['next'] is not None:
                    edge.obj.next = local_edges[edge.data['next']].obj
                if edge.data['prev'] is not None:
                    edge.obj.prev = local_edges[edge.data['prev']].obj
                if edge.data['face'] is not None:
                    edge.obj.face = local_faces[edge.data['face']].obj
        except Exception:
            logging.warning("Import Error for edge")

        try:
            #connect faces to their edges
            for face in local_faces.values():
                face.obj.edge_list = [local_edges[x].obj for x in face.data['edges']]
        except Exception:
            logging.warning("Import Error for face")

        #Now recalculate the quad tree as necessary
        self.calculate_quad_tree()

        #todo: pass the mapping back
        output_mapping['verts'] = {x.data['i'] : x.key for x in local_vertices.values()}
        output_mapping['edges'] = {x.data['i'] : x.key for x in local_edges.values()}
        output_mapping['faces'] = {x.data['i'] : x.key for x in local_faces.values()}
        return output_mapping

    @staticmethod
    def loadfile(filename):
        """ Create a DCEL from a saved pickle """
        if not isfile("{}.dcel".format(filename)):
            raise Exception("Non-existing filename to load into dcel")
        with open("{}.dcel".format(filename), 'rb') as f:
            dcel_data = pickle.load(f)
        the_dcel = DCEL()
        the_dcel.import_data(dcel_data)
        return the_dcel

    def savefile(self, filename):
        """ Save dcel data to a pickle """
        the_data = self.export_data()
        with open("{}.dcel".format(filename), 'wb') as f:
            pickle.dump(the_data, f)

