import numpy as np
from ..constants import EdgeE
from ...math import createLine, bezier1cp, bezier2cp, randomRad, rotMatrix
from .SampleSpec import SampleSpec
import cairo_utils.easings as easings

import logging as root_logger
import IPython
logging = root_logger.getLogger(__name__)

class CircleSample(SampleSpec):
    """ Sample Spec to Sample within a circle radius of data """

    def __init__(self, data, postFunction=None):
        super().__init__(data, postFunction)
        #setup default values

    def secondary_sample(self, core, target, data):
        """ Sample within a radius of each point """
        #for the xys, sample each vec_amnt times, along the vector, to distance
        assert("distance" in data)
        assert("vec_amnt" in data)
        sxys = np.zeros((1,data["vec_amnt"],2))
        if data["vec_amnt"] == 0:
            return sxys[1:]
        
        easing_1_fn = self.getEasing("easing_1", data)
        easing_2_fn = self.getEasing("easing_2", data)
        #distance
        easing_1 = easing_1_fn(np.random.random(data["vec_amnt"]))
        easing_1 *= data["distance"]
        
        easing_2 = easing_2_fn(np.linspace(0,1,data["vec_amnt"]))
        
        shuffler = "shuffle" in data and data["shuffle"]

        dist = easing_1
        rads = randomRad(shape=(data["vec_amnt"],))
        vec_array = np.array([np.array([1,0]) @ rotMatrix(x) for x in rads]).T
        vec_array *= dist
        vec_array *= easing_2
        vec_array = vec_array.T


        for a in core:
            if shuffler:
                rads = randomRad(shape=(data["vec_amnt"],))
                vec_array = np.array([np.array([1,0]) @ rotMatrix(x) for x in rads]).T
                vec_array *= dist
                vec_array *= easing_2
                vec_array = vec_array.T                
            sxys = np.row_stack((sxys, np.array([a + vec_array])))

        #primary, secondary
        return sxys[1:]

