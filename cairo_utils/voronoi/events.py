""" Events: The data representations of points and circles
    in the voronoi calculation
"""
##-- imports
from __future__ import annotations
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    List, Mapping, Match, MutableMapping, Optional, Sequence,
                    Set, Tuple, TypeVar, Union, cast)

import numpy as np

##-- end imports

CIRCLE_EVENTS = Enum("Circle Event Sides", "LEFT RIGHT")

@dataclass
class VEvent:
    """ The Base Class of events in fortunes algorithm """
    loc    : np.ndarray = field()
    step   : int        = field(default=-1)
    offset : float      = field(default=0)

    def y(self):
        """ Get the vertical position of the event """
        return self.loc[1]

    def __lt__(self, other):
        return (VEvent.offset - self.y()) < (VEvent.offset - other.y())

@dataclass
class SiteEvent(VEvent):
    """ Subclass for representing individual points / cell centres """
    face : Face = field(default=None)

    def __str__(self):
        return "Site Event: Loc: {}".format(self.loc)


@dataclass
class CircleEvent(VEvent):
    """ Subclass for representing the lowest point of a circle,
    calculated from three existing site events """

    #The node that will disappear
    sourceNode : Node   = field(default=None)
    #the breakpoint where it will disappear
    #vertex             == centre of circle, not lowest point
    vertex     : Vertex = field(default=None)
    left       : bool   = field(default=True)
    active     : bool   = field(default=True)

    def __post_init__(self, site_loc, sourceNode, voronoiVertex, left=True, i=None):
        if left and (CIRCLE_EVENTS.RIGHT in sourceNode.data
                     and sourceNode.data[CIRCLE_EVENTS.RIGHT].active):
            exception_text = "Trying to add a circle event to a taken left node: {} : {}"
            raise Exception(exception_text.format(sourceNode, sourceNode.data[CIRCLE_EVENTS.RIGHT]))
        elif not left and (CIRCLE_EVENTS.LEFT in sourceNode.data
                           and sourceNode.data[CIRCLE_EVENTS.LEFT].active):
            exception_text = "Trying to add a circle event to a taken right node: {} : {}"
            raise Exception(exception_text.format(sourceNode, sourceNode.data[CIRCLE_EVENTS.LEFT]))

        if left:
            sourceNode.data[CIRCLE_EVENTS.RIGHT] = self
        else:
            sourceNode.data[CIRCLE_EVENTS.LEFT] = self

    def __str__(self):
        return "Circle Event: {}, Node: {}, Left: {}, Added On Step: {}".format(self.loc,
                                                                                self.source,
                                                                                self.left,
                                                                                self.step)

    def deactivate(self):
        """ Deactivating saves on having to reheapify """
        self.active = False
