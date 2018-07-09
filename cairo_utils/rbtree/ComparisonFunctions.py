from enum import Enum
import math
import logging as root_logger
import IPython
import numpy as np
logging = root_logger.getLogger(__name__)

Directions = Enum('Directions', 'LEFT RIGHT')

#funcs take a node, a value, and data to contextualise

def default_comparison(a, b, compData):
    """ Standard smallest to largest comparison function """
    if a.value < b:
        return Directions.RIGHT

    return Directions.LEFT

def inverted_comparison(a, b, compData):
    """ Standard largest to smallest comparison function """
    if a.value < b:
        return Directions.LEFT
    return Directions.RIGHT

def default_equality(a, b, eqData):
    """ Standard Equality test """
    return a.value == b

def arc_equality(a, b, eqData):
    #todo: return true if  a.pred|a < b < a|a.succ
    raise Exception("Unimplemented")

def arc_comparison(a, b, compData):
    """ Function to compare an arc and xposition
    Used in Beachline/Voronoi """
    pred = a.getPredecessor()
    succ = a.getSuccessor()
    logging.debug("Pred: {}, Succ: {}".format(pred,succ))
    pred_intersect = None
    succ_intersect = None
    the_range = [-math.inf,math.inf]
    if pred == None and succ == None: #Base case: single arc
        logging.debug("Single Arc: {}".format(a.value))
        if b < a.value.fx:
            return Directions.LEFT
        else:
            return Directions.RIGHT

    pred_self = False
    self_succ = False
            
    if pred != None:
        pred_intersect = a.value.intersect(pred.value)
        pred_int = pred_intersect.astype(dtype=np.int)
        pred_above_self = a.value.fy < pred.value.fy
        logging.debug("Comparing x:{:.1f} to pred_intersect: {}".format(b, pred_int))
        logging.debug("Pred above self: {}".format(pred_above_self))
        if pred_intersect is not None and len(pred_intersect) == 1:
            if b < pred_intersect[0,0]:
                pred_self = True
        elif pred_intersect is not None and len(pred_intersect) == 2:
            if pred_above_self:
                if b < pred_intersect[0,0]:
                    pred_self = True
            else:
                if b < pred_intersect[1,0]:
                    pred_self = True
                
    if succ != None:
        succ_intersect = succ.value.intersect(a.value)
        succ_int = succ_intersect.astype(dtype=np.int)
        succ_above_self = a.value.fy < succ.value.fy
        logging.debug("Comparing b:{:.1f} to succ_intersect: {}".format(b, succ_int))
        logging.debug("Succ above a: {}".format(succ_above_self))
        if succ_intersect is not None and len(succ_intersect) == 1:
            if b > succ_intersect[0,0]:
                self_succ = True
        elif succ_intersect is not None and len(succ_intersect) == 2:
            if succ_above_self:
                if succ_intersect[1,0] < b:
                    self_succ = True
            else:
                if succ_intersect[0,0] < b:
                    self_succ = True

    logging.debug("pred_self: {}".format(pred_self))
    logging.debug("self_succ: {}".format(self_succ))

    
    
    if pred_self:
        return Directions.LEFT
    if self_succ:
        return Directions.RIGHT
    
    return Directions.LEFT

