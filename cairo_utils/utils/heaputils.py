"""
Utilities for using heapq
"""
##-- imports
from __future__ import annotations
import heapq
import logging as root_logger
from dataclasses import InitVar, dataclass, field
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    List, Mapping, Match, MutableMapping, Optional, Sequence,
                    Set, Tuple, TypeVar, Union, cast)

##-- end imports

logging = root_logger.getLogger(__name__)

def pop_while_same(heap):
    """ Pop while the head is equal to the first value poppped """
    assert(all([isinstance(x, HeapWrapper) for x in heap]))
    first_vert, first_edge = heapq.heappop(heap).unwrap()
    if first_edge is None:
        return (first_vert, [])

    collected = (first_vert, [first_edge])
    count = 1
    while bool(heap) and heap[0].ordinal == first_vert:
        data = heapq.heappop(heap).data
        if data is not None:
            collected[1].append(data)
            count += 1
    assert(len(collected[1]) == count)
    return collected

@dataclass
class HeapWrapper:
    """ Utility to wrap an ordinal with data to use in the heap """
    ordinal : int
    data    : Any
    desc    : str = field(default=None)

    def __lt__(self, other):
        assert(isinstance(other, HeapWrapper))
        return self.ordinal < other.ordinal

    def unwrap(self):
        """ Unwrap the data """
        return (self.ordinal, self.data)

    def __repr__(self):
        if self.desc is None:
            return "{} - {}".format(self.ordinal, repr(self.data))
        else:
            return "{} - {} : {}".format(self.ordinal, self.desc, repr(self.data))
