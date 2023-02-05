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


class VoronoiGraph:

    @staticmethod
    def build(data=None, rerun=False, random=None):
        """ Create a graph of initial random sites """
        logging.debug("Initialising graph")
        self.reset()

        if random is None:
            random = base_random.random

        values = data
        if values is None and not rerun:
            values = self.load_graph()

        assert(values is None or isinstance(values, np.ndarray))
        #create a (n, 2) array of coordinates for the sites, if no data has been loaded
        if values is None:
            logging.debug("Generating values")
            for _ in range(self.node_size):
                rnd_amnt = random((1, 2))
                #scale the new site
                scaler = self.bbox.reshape((2, 2)).transpose()
                new_site = scaler[:, 0] + (rnd_amnt * (scaler[:, 1] - scaler[:, 0]))
                if values is None:
                    values = new_site
                else:
                    values = np.row_stack((values, new_site))

        #setup the initial site events:
        used_coords = []
        for site in values:
            #Avoid duplications:
            if (site[0], site[1]) in used_coords:
                logging.warning("Skipping Duplicate: {}".format(site))
                continue
            #Create an empty face for the site
            future_face = self.dcel.newFace(site, data=BASE_VORONOI_FACE_DATA)
            event = SiteEvent(site, face=future_face)
            heapq.heappush(self.events, event)
            self.sites.append(event)
            used_coords.append((site[0], site[1]))

        #Save the nodes
        if not rerun:
            self.save_graph(values)
        return values
