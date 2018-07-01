import numpy as np
from functools import partial
from ..constants import EdgeE
from ...math import createLine, bezier1cp, bezier2cp, rotMatrix, randomRad
from .SampleSpec import SampleSpec
from .. import Face, HalfEdge, Vertex

import cairo_utils.easings as easings

import logging as root_logger
import IPython
logging = root_logger.getLogger(__name__)

rotMul = lambda r: np.array([1,0]) @ r

class AngleSample(SampleSpec):
    """ Samples the given data in an angle relative to the data """

    def __init__(self, data, postFunction=None):
        super().__init__(data, postFunction)
        #setup default values

    def secondary_sample(self, core, target, data):
        """ Sample in the direction of an angle  """
        assert("rads" in data or "radRange" in data)
        assert("vec_amnt" in data)
        assert("distance" in data)
        
        shuffler = "shuffle" in data and data["shuffle"]
        inc_amount = ("inc" in data and data["inc"]) or 0
        incRange = None
        if "incRange" in data:
            incRange = randomRad(min=data["incRange"][0],
                                 max=data["incRange"][1],
                                 shape=(data["vec_amnt"],))
        
        easing_1_fn = self.getEasing("easing_1", data)
        easing_1 = easing_1_fn(np.random.random(data["vec_amnt"]))

        initial = np.array([1,0]) * data["distance"]
        rads = None
        if "rads" in data:
            rads = np.repeat(data["rads"], data["vec_amnt"])

        else:
            rads = randomRad(min=data["radRange"][0],
                             max=data["radRange"][1],
                             shape=(data["vec_amnt"],))

        vec_array = np.array([initial @ rotMatrix(x) for x in rads]).T
        vec_array *= easing_1
        vec_array = vec_array.T
        
        sxys = np.zeros((1,data["vec_amnt"],2))
        for a in core:
            if shuffler and "radRange" in data:
                rads = randomRad(min=data["radRange"][0],
                                 max=data["radRange"][1],
                                 shape=(data["vec_amnt"],))
                vec_array = np.array([initial @ rotMatrix(x) for x in rads]).T
                vec_array *= easing_1
                vec_array = vec_array.T
            elif incRange is not None:
                rads += incRange
                vec_array = np.array([initial @ rotMatrix(x) for x in rads]).T
                vec_array *= easing_1
                vec_array = vec_array.T
            elif inc_amount != 0:
                rads += inc_amount
                vec_array = np.array([initial @ rotMatrix(x) for x in rads]).T
                vec_array *= easing_1
                vec_array = vec_array.T
                
            sxys = np.row_stack((sxys, np.array([a + vec_array])))
        #primary, secondary
        return sxys[1:]

