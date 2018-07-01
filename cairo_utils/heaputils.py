"""
Utilities for using heapq
"""
import heapq
import logging as root_logger
import IPython
logging = root_logger.getLogger(__name__)

class HeapWrapper:
    """ Utility to wrap an ordinal with data to use in the heap """
    def __init__(self, ord, data, desc=None):
        self.ord = ord
        self.data = data
        self.desc = desc

    def __lt__(self, other):
        assert(isinstance(other, HeapWrapper))
        return self.ord < other.ord

    def unwrap(self):
        return (self.ord, self.data)

    def __repr__(self):
        if self.desc is None:
            return "{} - {}".format(self.ord, repr(self.data))
        else:
            return "{} - {} : {}".format(self.ord, self.desc, repr(self.data))

def pop_while_same(heap):
    """ Pop while the head is equal to the first value poppped """
    assert(all([isinstance(x, HeapWrapper) for x in heap]))
    first_vert, first_edge = heapq.heappop(heap).unwrap()
    if first_edge is None:
        return (first_vert, [])
    
    collected = (first_vert, [first_edge])
    count = 1
    while bool(heap) and heap[0].ord == first_vert:
        data = heapq.heappop(heap).data
        if data is not None:
            collected[1].append(data)
            count += 1
    assert(len(collected[1]) == count)
    return collected

