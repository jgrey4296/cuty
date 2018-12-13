"""
Provides a sampler set at a specific angle
"""
import logging as root_logger
import numpy as np
from ...math import rotation_matrix, random_radian
from .sample_spec import SampleSpec

logging = root_logger.getLogger(__name__)

ROT_MUL = lambda r: np.array([1, 0]) @ r

class AngleSample(SampleSpec):
    """ Samples the given data in an angle relative to the data """

    def __init__(self, data, post_function=None):
        super().__init__(data, post_function)
        #setup default values

    def secondary_sample(self, core, target, data):
        """ Sample in the direction of an angle  """
        assert("rads" in data or "radRange" in data)
        assert("vec_amnt" in data)
        assert("distance" in data)

        shuffler = "shuffle" in data and data["shuffle"]
        inc_amount = (data['inc'] if 'inc' in data else 0)
        inc_range = None
        if "inc_range" in data:
            inc_range = random_radian(min_v=data["inc_range"][0],
                                      max_v=data["inc_range"][1],
                                      shape=(data["vec_amnt"]))

        easing_1_fn = self.get_easing("easing_1", data)
        easing_1 = easing_1_fn(np.random.random(data["vec_amnt"]))

        initial = np.array([1, 0]) * data["distance"]
        rads = None
        if "rads" in data:
            rads = np.repeat(data["rads"], data["vec_amnt"])

        else:
            rads = random_radian(min_v=data["radRange"][0],
                                 max_v=data["radRange"][1],
                                 shape=(data["vec_amnt"]))

        vec_array = np.array([initial @ rotation_matrix(x) for x in rads]).T
        vec_array *= easing_1
        vec_array = vec_array.T

        sxys = np.zeros((1, data["vec_amnt"], 2))
        for a in core:
            if shuffler and "radRange" in data:
                rads = random_radian(min_v=data["radRange"][0],
                                     max_v=data["radRange"][1],
                                     shape=(data["vec_amnt"]))
                vec_array = np.array([initial @ rotation_matrix(x) for x in rads]).T
                vec_array *= easing_1
                vec_array = vec_array.T
            elif inc_range is not None:
                rads += inc_range
                vec_array = np.array([initial @ rotation_matrix(x) for x in rads]).T
                vec_array *= easing_1
                vec_array = vec_array.T
            elif inc_amount != 0:
                rads += inc_amount
                vec_array = np.array([initial @ rotation_matrix(x) for x in rads]).T
                vec_array *= easing_1
                vec_array = vec_array.T

            sxys = np.row_stack((sxys, np.array([a + vec_array])))
        #primary, secondary
        return sxys[1:]
