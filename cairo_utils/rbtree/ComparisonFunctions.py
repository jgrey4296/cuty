from enum import Enum
import logging as root_logger
logging = root_logger.getLogger(__name__)

Directions = Enum('Directions', 'LEFT RIGHT CENTRE')

#funcs take a node, a value, and data to contextualise

def default_comparison(a, b, compData):
    """ Standard smallest to largest comparison function """
    if a.value < b:
        return Directions.RIGHT
    elif b < a.value:
        return Directions.LEFT
    return Directions.CENTRE

def inverted_comparison(a, b, compData):
    """ Standard largest to smallest comparison function """
    if a.value < b:
        return Directions.LEFT
    elif b < a.value:
        return Directions.RIGHT
    return Directions.CENTRE

def default_equality(a, b, eqData):
    """ Standard Equality test """
    return a.value == b

def arc_comparison(a, b, compData):
    """ Function to compare two arcs. 
    Used in Beachline/Voronoi """
    pred = a.get_predecessor()
    succ = a.get_successor()
    logging.debug("Pred: {}, Succ: {}".format(pred,succ))
    pred_intersect = None
    succ_intersect = None
    the_range = [-math.inf,math.inf]
    if pred == NilNode and succ == NilNode: #Base case: single arc
        logging.debug("Single Arc: {}".format(a.value))
        if b < a.value.fx:
            return Directions.LEFT
        else:
            return Directions.RIGHT

    pred_self = Directions.CENTRE
    self_succ = Directions.CENTRE
            
    if pred != NilNode:
        pred_intersect = a.value.intersect(pred.value)
        pred_int = pred_intersect.astype(dtype=np.int)
        pred_above_self = a.value.fy < pred.value.fy
        logging.debug("Comparing x:{:.1f} to pred_intersect: {}".format(b, pred_int))
        logging.debug("Pred above self: {}".format(pred_above_self))
        if pred_intersect is None:
            #pass through, no intersection
            pred_self = Directions.CENTRE
        elif len(pred_intersect) == 1:
            if b < pred_intersect[0,0]:
                pred_self = Directions.LEFT
        elif len(pred_intersect) == 2:
            if pred_above_self:
                if b < pred_intersect[0,0]:
                    pred_self = Directions.LEFT
            else:
                if b < pred_intersect[1,0]:
                    pred_self = Directions.LEFT
                
    if succ != NilNode:
        succ_intersect = succ.value.intersect(a.value)
        succ_int = succ_intersect.astype(dtype=np.int)
        succ_above_self = a.value.fy < succ.value.fy
        logging.debug("Comparing b:{:.1f} to succ_intersect: {}".format(b, succ_int))
        logging.debug("Succ above a: {}".format(succ_above_self))
        if succ_intersect is None:
            #pass through
            self_succ = Directions.CENTRE
        elif len(succ_intersect) == 1:
            if b > succ_intersect[0,0]:
                self_succ = Directions.RIGHT
        elif len(succ_intersect) == 2:
            if succ_above_self:
                if succ_intersect[1,0] < b:
                    self_succ = Directions.RIGHT
            else:
                if succ_intersect[0,0] < b:
                    self_succ = Directions.RIGHT

    logging.debug("pred_self: {}".format(pred_self))
    logging.debug("self_succ: {}".format(self_succ))
                        
    if pred_self is Directions.CENTRE and self_succ is Directions.CENTRE:
        return pred_self
    if pred_self is Directions.LEFT:
        return pred_self
    if self_succ is Directions.RIGHT:
        return self_succ
    
    return Directions.CENTRE

