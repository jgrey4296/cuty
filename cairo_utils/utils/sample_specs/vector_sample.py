"""
Provides a sampler that translates by a fixed vector
"""
##-- imports
from __future__ import annotations
import logging as root_logger
import numpy as np
from .sample_spec import SampleSpec

##-- end imports

logging = root_logger.getLogger(__name__)

class VectorSample(SampleSpec):
    """ An implementation of a sample spec to translate by a vector """

    def __init__(self, data, post_function=None):
        super().__init__(data, post_function)
        #setup default values

    def secondary_sample(self, core, target, data, random=None):
        """ for the xys, sample each vec_amnt times,
        along the vector, to distance
        """
        assert("vector" in data)
        assert("distance" in data)
        assert("sample_amnt" in data)
        assert("vec_amnt" in data)

        if random is None:
            random = np.random.random
        easing_1_fn = self.get_easing("easing_1", data)
        easing_1 = easing_1_fn(random(data["vec_amnt"]))

        vec_array = np.repeat(data["vector"], data["vec_amnt"]).reshape((2, -1))
        vec_array *= data["distance"]
        vec_array *= easing_1
        vec_array = vec_array.T

        sxys = np.zeros((1, data["vec_amnt"], 2))
        for a in core:
            sxys = np.row_stack((sxys, np.array([a + vec_array])))

        #primary, secondary
        return sxys[1:]
