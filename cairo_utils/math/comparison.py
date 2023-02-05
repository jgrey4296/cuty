"""
Provides default comparison functions to be used in the RBTree
Including standard <, > and ==,
along with arc comparisons for beachline use
"""
##-- imports
from __future__ import annotations

from enum import Enum
import logging as root_logger

##-- end imports

logging = root_logger.getLogger(__name__)
Directions = Enum('Directions', 'LEFT RIGHT')

#funcs take a node, a value, and data to contextualise

def default_comparison(a, b, comp_data):
    """ Standard smallest to largest comparison function """
    if a.value < b:
        return Directions.RIGHT
    return Directions.LEFT

def inverted_comparison(a, b, comp_data):
    """ Standard largest to smallest comparison function """
    if a.value < b:
        return Directions.LEFT
    return Directions.RIGHT

def default_equality(a, b, eq_data):
    """ Standard Equality test """
    return a.value == b


class ArcCmp:
    def arc_equality(a, b, eq_data):
        """ Compare two arcs for equality """
        #return true if  a.pred|a < b < a|a.succ
        l_inter, r_inter = __arc_intersects(a, b, eq_data)
        result = False
        if l_inter is not None and r_inter is not None:
            result = l_inter < b < r_inter
        elif r_inter is not None:
            result = b < r_inter
        elif l_inter is not None:
            result = l_inter < b
        return result

    def arc_comparison(a, b, comp_data):
        """ Function to compare an arc and xposition
        Used in Beachline/Voronoi """
        l_inter, r_inter = __arc_intersects(a, b, comp_data)
        pred_self = False
        self_succ = False

        if l_inter is None and r_inter is None: #Base case: single arc
            if b < a.value.fx:
                return Directions.LEFT
            else:
                return Directions.RIGHT

        if l_inter is not None:
            if b < l_inter:
                pred_self = True
        if r_inter is not None:
            if r_inter < b:
                self_succ = True

        if pred_self:
            return Directions.LEFT
        if self_succ:
            return Directions.RIGHT

        return Directions.LEFT

    def __arc_intersects(a, b, comp_data):
        """ Internal function to test if two arcs intersect """
        pred = a.getPredecessor()
        succ = a.getSuccessor()
        pred_intersect = None
        succ_intersect = None
        pred_intersect_out = None
        succ_intersect_out = None
        pred_above_self = None
        succ_above_self = None

        if pred is not None:
            pred_intersect = a.value.intersect(pred.value)
            pred_above_self = a.value.fy < pred.value.fy

        if succ is not None:
            succ_intersect = succ.value.intersect(a.value)
            succ_above_self = a.value.fy < succ.value.fy

        if pred_intersect is not None and len(pred_intersect) == 1:
            pred_intersect_out = pred_intersect[0, 0]
        elif pred_intersect is not None and len(pred_intersect) == 2:
            if pred_above_self:
                pred_intersect_out = pred_intersect[0, 0]
            else:
                pred_intersect_out = pred_intersect[1, 0]

        if succ_intersect is not None and len(succ_intersect) == 1:
            succ_intersect_out = succ_intersect[0, 0]
        elif succ_intersect is not None and len(succ_intersect) == 2:
            if succ_above_self:
                succ_intersect_out = succ_intersect[1, 0]
            else:
                succ_intersect_out = succ_intersect[0, 0]

        return (pred_intersect_out, succ_intersect_out)


class LineCmp:
    def line_cmp(a, b, cd):
        """ Line comparison to be used in the status tree """
        #Is horizontal:
        a_hor = a.value.is_flat()
        b_hor = b.is_flat()
        #---
        logging.debug("Comparison: {} - {}".format(a.value.index, b.index))
        logging.debug("Flat:       {} - {}".format(a_hor, b_hor))
        y = cd['y']
        if not (a_hor or b_hor):
            y += cd['nudge']
        a_ranges = a.value.get_ranges()
        b_ranges = b.get_ranges()

        a_val = a.value(y=y)[0]
        b_val = b(y=y)[0]
        if b_hor:
            b_val = min(max(cd['x'], b_ranges[0, 0]), b_ranges[0, 1])
        if a_hor:
            a_val = min(max(cd['x'], a_ranges[0, 0]), a_ranges[0, 1])

        logging.debug("Values: {} - {}".format(a_val, b_val))

        if a_val <= b_val:
            return Directions.RIGHT
        return Directions.LEFT

    def line_eq(a, b, cd):
        """ Compare two lines against each other """
        #pylint: disable=unused-argument
        return a.value == b

    def line_cmp_vert(a, b, cd):
        """ Compare a line against a point """
        result = Directions.LEFT
        if a.value.is_flat():
            a_val = cd['x']
        else:
            a_val = a.value(y=cd['y'])[0]
        logging.debug("VERT a_val{}: {}  b_val: {}".format(a.value.index, a_val, b[0]))

        if a_val <= b[0]:
            result = Directions.RIGHT
        return result

    def line_eq_vert(a, b, cd):
        """ test for equality between a line and a point """
        return np.allclose(a.value(y=cd['y']), b)
