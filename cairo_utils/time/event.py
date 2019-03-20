"""
Events express when a value holds in time
"""
from .arc import Arc
from fractions import Fraction
import logging as root_logger

logging = root_logger.getLogger(__name__)

class Event:
    """ A Value active during a timespan """

    def __init__(self, a, b, value_is_pattern=False):
        assert(isinstance(a, Arc))
        self.arc = a.copy()
        self.value = b
        self.value_is_pattern = value_is_pattern

    def __call__(self, count):
        """ Get a list of events given a time """
        if count in self.arc:
            if self.value_is_pattern:
                return self.value(count - self.arc.start)
            else:
                return [self]
        return []

    def base(self):
        """ Get all fractions used in this event """
        time_list = self.arc.pair()
        if self.value_is_pattern:
            time_list += [x - self.arc.start for x in self.value.base()]
        return set(time_list)

    def key(self):
        """ Get the start of the event, for sorting """
        return self.arc.start

    def __contains__(self, other):
        return other in self.arc

    def __repr__(self):
        return "{} :: {}".format(str(self.value), str(self.arc))

    def print_flip(self, start=True):
        """ Get a string describing the event's entry/exit status """
        fmt_str ="⤒{} "
        if not start:
            fmt_str = "{}⤓"
        return fmt_str.format(str(self.value))
