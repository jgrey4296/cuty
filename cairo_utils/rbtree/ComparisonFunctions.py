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
    #return true if  a.pred|a < b < a|a.succ
    l_inter, r_inter = __arc_intersects(a,b,eqData)
    result = False
    if l_inter is not None and r_inter is not None:
        result = l_inter < b < r_inter
    elif r_inter is not None:
        result = b < r_inter
    elif l_inter is not None:
        result = l_inter < b
    return result

def arc_comparison(a, b, compData):
    """ Function to compare an arc and xposition
    Used in Beachline/Voronoi """
    l_inter, r_inter = __arc_intersects(a,b,compData)
    pred_self = False
    self_succ = False

    if l_inter == None and r_inter == None: #Base case: single arc
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

def __arc_intersects(a, b, compData):
    pred = a.getPredecessor()
    succ = a.getSuccessor()
    pred_intersect = None
    succ_intersect = None
    pred_intersect_out = None
    succ_intersect_out = None
    pred_above_self = None
    succ_above_self = None
    
    if pred != None:
        pred_intersect = a.value.intersect(pred.value)
        pred_above_self = a.value.fy < pred.value.fy
                
    if succ != None:
        succ_intersect = succ.value.intersect(a.value)
        succ_above_self = a.value.fy < succ.value.fy

    if pred_intersect is not None and len(pred_intersect) == 1:
        pred_intersect_out = pred_intersect[0,0]
    elif pred_intersect is not None and len(pred_intersect) == 2:
        if pred_above_self:
            pred_intersect_out = pred_intersect[0,0]
        else:
            pred_intersect_out = pred_intersect[1,0]

    if succ_intersect is not None and len(succ_intersect) == 1:
        succ_intersect_out = succ_intersect[0,0]
    elif succ_intersect is not None and len(succ_intersect) == 2:
            if succ_above_self:
                succ_intersect_out = succ_intersect[1,0]
            else:
                succ_intersect_out = succ_intersect[0,0]
        
    return (pred_intersect_out, succ_intersect_out)
