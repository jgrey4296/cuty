"""
Events express when a value holds in time
"""
##-- imports
from __future__ import annotations
from .arc import Arc
from fractions import Fraction
import logging as root_logger

##-- end imports

logging = root_logger.getLogger(__name__)

class Event:
    """ A Value active during a timespan """

    def __init__(self, a, b, value_is_pattern=False, params=None):
        assert(isinstance(a, Arc))
        self.arc              = a.copy()
        self.values           = b
        self.parameters       = params or {}
        self.value_is_pattern = value_is_pattern

    def __call__(self, count, just_values=False, rand_s=None):
        """ Get a list of events given a time """
        match (count in self.arc), self.value_is_pattern:
            case True, True:
                return self.values(count - self.arc.start, just_values)
            case False, True:
                return [self]
            case _:
                return []

    def __contains__(self, other):
        return other in self.arc

    def __repr__(self):
        return f"{self.values} :: {self.arc}"


    def base(self):
        """ Get all fractions used in this event """
        time_list = self.arc.pair()
        if self.value_is_pattern:
            time_list += [x - self.arc.start for x in self.values.base()]
        return set(time_list)

    def key(self):
        """ Get the start of the event, for sorting """
        return self.arc.start

    def print_flip(self, start=True):
        """ Get a string describing the event's entry/exit status """
        return f"⤒{self.values} " if start else f"{self.values}⤓"

    def __getitem__(self, val):
        """ event[x] """
        return self.parameters[val]
