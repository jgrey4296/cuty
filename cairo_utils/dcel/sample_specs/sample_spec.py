"""
Provides the superclass to implement a sampler
"""
import logging as root_logger
import numpy as np
import cairo_utils.easings as easings
from ...constants import SMALL_RADIUS, VERTEX
from ..constants import EdgeE
from ...umath import create_line, bezier1cp, bezier2cp, get_distance
from ...drawing import draw_circle
from .. import Face, HalfEdge, Vertex



logging = root_logger.getLogger(__name__)


class SampleSpec:
    """ Abstract Spec to describe the sampling transform
      from DCEL -> Point Cloud
    """

    def __init__(self, data, postFunction=None):
        self.data = data
        if postFunction is None:
            postFunction = default_post
        assert(callable(postFunction))
        self.post_fn = postFunction

    def __repr__(self):
        return "SampleSpec"

    def __call__(self, ctx, target, data=None):
        #sample
        complete_data = {}
        complete_data.update(self.data)
        if data is not None:
            complete_data.update(data)
        core_positions = self.primary_sample(target, complete_data)
        secondary_positions = self.secondary_sample(core_positions, target, complete_data)
        #post
        core_details, secondary_details = self.post_fn(core_positions,
                                                       secondary_positions,
                                                       target,
                                                       complete_data)
        #draw
        self.draw_point_cloud(ctx, core_details)
        for line in secondary_details:
            self.draw_point_cloud(ctx, line)

    def primary_sample(self, target, data):
        """ The method to override """
        if isinstance(target, Vertex):
            return self.primary_vertex(target, data)
        elif isinstance(target, HalfEdge):
            return self.primary_halfedge(target, data)
        elif isinstance(target, Face):
            return self.primary_face(target, data)
        else:
            raise Exception("TODO: Step Down Automatically on Missing Sample Functions")


    def primary_vertex(self, target, data):
        """ Samples a single vertex """
        return np.array([target.toArray()])


    def primary_halfedge(self, target, data):
        """ Sample the halfEdge to start with """
        assert("sample_amnt" in data)
        #Primary
        sample_amount = data["sample_amnt"]
        pxys = np.zeros((1, 2))

        #calculate the actual line
        logging.info("Sampling a line")
        #start and end
        if EdgeE.BEZIER in target.data:
            for b in target.data[EdgeE.BEZIER]:
                num_points = get_distance(b[0], b[-1]) * sample_amount
                if len(b) == 3:
                    new_points = bezier1cp(*b, num_points)
                else:
                    assert(len(b) == 4)
                    new_points = bezier2cp(*b, num_points)
                pxys = np.row_stack((pxys, new_points))
        else:
            #straight line
            #TODO CORRECT THIS
            target_array = target.to_array()
            num_points = get_distance(target_array[0], target_array[1]) * sample_amount
            pxys = np.row_stack((pxys, create_line(*target_array[0],
                                                   *target_array[1],
                                                   num_points)))
        assert(pxys is not None)
        return pxys[1:]


    def primary_face(self, target, data):
        """ Samples a single Face """
        #create sample amnt of points within the bounds of the face
        if "fill" in data:
            #choose points from within the polygon
            return np.array([])
        elif "stroke" in data:
            #choose points along the edges
            return np.array([])
        elif "corner" in data:
            #choose points at the corners of the polygon
            return np.array([])
        else:
            raise Exception("Unrecognised face sample instruction")

    def secondary_sample(self, core, target, data):
        """ The method to override """
        raise Exception("SampleSpec.secondary_sample should be overriden")

    def draw_point_cloud(self, ctx, xyrcs):
        """ Expect an array of [x, y, radius, r, g, b, a] """
        assert(xyrcs.shape[1] == 7)
        for line in xyrcs:
            ctx.set_source_rgba(*line[3:])
            draw_circle(ctx, *line[:2], line[2])

    def get_easing(self, key, data):
        """ lookup an easing function   """
        try:
            easing_data = data[key]
            easing_fn = easings.lookup(easing_data[0])
            if len(easing_data) > 2:
                codomain = easing_data[2]
            else:
                codomain = easings.CODOMAIN.FULL
            if len(easing_data) > 3:
                quantize = easing_data[3]
            else:
                quantize = 0

            if quantize != 0:
                easing_lambda = lambda xs: easings.quantize(easing_fn(xs,
                                                                      easing_data[1],
                                                                      codomain_e=codomain),
                                                            q=quantize)
            else:
                easing_lambda = lambda xs: easing_fn(xs, easing_data[1], codomain_e=codomain)
            return (easing_lambda)
        except Exception as e:
            logging.warning(e)
            easing_fn = easings.lookup(easings.ENAMES[0])
            return (lambda xs: easing_fn(xs, 1, codomain_e=easings.CODOMAIN.FULL))




def default_post(core, secondary, target, data):
    """ add default radius and colours to points """
    assert(len(core.shape) == 2)
    assert(len(secondary.shape) == 3)
    #add radius and then colours
    pradius = SMALL_RADIUS
    pcolour = VERTEX
    if 'radius' in data:
        pradius = data['radius']
    if 'colour' in data:
        pcolour = data['colour']

    radi = np.array(pradius).repeat(len(core))
    colours = pcolour.repeat(len(core)).reshape((4, -1)).T
    core_mode = np.column_stack((core, radi, colours))

    sec_len = len(secondary[0])
    sec_radi = np.array(pradius).repeat(sec_len)
    sec_colours = pcolour.repeat(sec_len).reshape((4, -1)).T
    sec_mod = np.zeros((1, sec_len, 7))
    for a in secondary:
        b = np.column_stack((a, sec_radi, sec_colours))
        sec_mod = np.row_stack((sec_mod, np.array([b])))

    return (core_mode, sec_mod[1:])
