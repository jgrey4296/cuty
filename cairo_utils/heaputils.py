"""
Utilities for using heapq
"""
import heapq
import logging as root_logger
logging = root_logger.getLogger(__name__)

def pop_while_same(heap):
    """ Pop while the head is equal to the first value poppped """
    first_vert, first_edge = heapq.heappop(heap)
    collected = (first_vert, [first_edge])
    while bool(heap) and heap[0][0] == first_vert:
        collected[1].append(heapq.heappop(heap)[1])

    return collected

