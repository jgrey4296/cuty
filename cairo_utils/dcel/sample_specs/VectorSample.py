import numpy as np
from ..constants import EdgeE
from ...math import createLine, bezier1cp, bezier2cp
from .SampleSpec import SampleSpec
import cairo_utils.easings as easings

import logging as root_logger
import IPython
logging = root_logger.getLogger(__name__)



class VectorSample(SampleSpec):
    """ An implementation of a sample spec to translate by a vector """

    def __init__(self, data, postFunction=None):
        super().__init__(data, postFunction)
        #setup default values

        
    def primary_sample(self, target, data):
        """ The method to override """
        assert("vector" in data)
        assert("distance" in data)
        assert("sample_amnt" in data)
        assert("vec_amnt" in data)
        #Primary
        pxys = None
        sample_amount = data["sample_amnt"]
        vec_array = np.repeat(data["vector"], data["vec_amnt"]).reshape((2,-1))
        vec_array *= data["distance"]
        vec_array = vec_array.T
    
        #calculate the actual line
        logging.info("Sampling a line")
        #start and end
        se = target.toArray()
        if EdgeE.BEZIER in target.data:
            #get control points, calculate
            cp1, cp2 = target.data[EdgeE.BEZIER]
            if cp2 is None:
                pxys = bezier1cp(se[0], cp1, se[1], sample_amount)
            else:
                pxys = bezier2cp(se[0], cp1, cp2, se[1], sample_amount)
        else:
            #straight line
            pxys = createLine(*se[0], *se[1], sample_amount)
        assert(pxys is not None)
        return pxys

    def secondary_sample(self, core, target, data):
        #for the xys, sample each vec_amnt times, along the vector, to distance
        vec_array = np.repeat(data["vector"], data["vec_amnt"]).reshape((2,-1))
        vec_array *= data["distance"]
        vec_array *= easings.pow_abs_curve(np.linspace(0,1,data["vec_amnt"]), 1)
        vec_array = vec_array.T

        sxys = np.zeros((1,data["vec_amnt"],2))
        for a in core:
            sxys = np.row_stack((sxys, np.array([a + vec_array])))

        #primary, secondary
        return sxys[1:]

