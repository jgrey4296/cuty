"""
Provides a sampler operating in a random circle
"""
import logging as root_logger
import numpy as np
from ...umath import random_radian, rotation_matrix
from .sample_spec import SampleSpec

logging = root_logger.getLogger(__name__)

class CircleSample(SampleSpec):
    """ Sample Spec to Sample within a circle radius of data """

    def __init__(self, data, postFunction=None):
        super().__init__(data, postFunction)
        #setup default values

    def secondary_sample(self, core, target, data, random=None):
        """ Sample within a radius of each point """
        #for the xys, sample each vec_amnt times, along the vector, to distance
        assert("distance" in data)
        assert("vec_amnt" in data)
        if random is None:
            random = np.random.random
        sxys = np.zeros((1, data["vec_amnt"], 2))
        if data["vec_amnt"] == 0:
            return sxys[1:]

        easing_1_fn = self.get_easing("easing_1", data)
        easing_2_fn = self.get_easing("easing_2", data)
        #distance
        easing_1 = easing_1_fn(random(data["vec_amnt"]))
        easing_1 *= data["distance"]

        easing_2 = easing_2_fn(np.linspace(0, 1, data["vec_amnt"]))

        shuffler = "shuffle" in data and data["shuffle"]

        dist = easing_1
        rads = random_radian(shape=(data["vec_amnt"],), random=random)
        vec_array = np.array([np.array([1, 0]) @ rotation_matrix(x) for x in rads]).T
        vec_array *= dist
        vec_array *= easing_2
        vec_array = vec_array.T


        for a in core:
            if shuffler:
                rads = random_radian(shape=(data["vec_amnt"],), random=random)
                vec_array = np.array([np.array([1, 0]) @ rotation_matrix(x) for x in rads]).T
                vec_array *= dist
                vec_array *= easing_2
                vec_array = vec_array.T
            sxys = np.row_stack((sxys, np.array([a + vec_array])))

        #primary, secondary
        return sxys[1:]
