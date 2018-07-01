import numpy as np
import IPython
from ...constants import SMALL_RADIUS, VERTEX
from ..constants import EdgeE
from ...math import createLine, bezier1cp, bezier2cp, get_distance
from ...drawing import drawRect, drawCircle, clear_canvas, drawText
from .. import Face, HalfEdge, Vertex
import cairo_utils.easings as easings

import logging as root_logger
logging = root_logger.getLogger(__name__)


class SampleSpec:
    """ Abstract Spec to describe the sampling transform 
      from DCEL -> Point Cloud
    """

    def __init__(self, data, postFunction=None):
        self.data = data
        if postFunction is None:
            postFunction = defaultPost
        assert(callable(postFunction))
        self.postFn = postFunction

    def __repr__(self):
        return "SampleSpec"
        
    def __call__(self, ctx, target, data=None):
        #sample
        completeData = {}
        completeData.update(self.data)
        if data is not None:
            completeData.update(data)
        corePositions = self.primary_sample(target,completeData)
        secondaryPositions = self.secondary_sample(corePositions, target, completeData)
        #post
        coreDetails, secondaryDetails = self.postFn(corePositions, secondaryPositions, target, completeData)
        #draw
        self.draw_point_cloud(ctx, coreDetails)
        for line in secondaryDetails:
            self.draw_point_cloud(ctx, line)

    def primary_sample(self, target, data):
        """ The method to override """
        if isinstance(target, Vertex):
            return self.primary_vertex(target, data)
        elif isinstance(target, HalfEdge):
            return self.primary_halfEdge(target, data)
        elif isinstance(target, Face):
            return self.primary_face(target,data)
        else:
            raise Exception("TODO: Step Down Automatically on Missing Sample Functions")
        
        
    def primary_vertex(self, target, data):
        """ Samples a single vertex """
        return np.array([target.toArray()])
        

    def primary_halfEdge(self, target, data):
        """ Sample the halfEdge to start with """
        assert("sample_amnt" in data)
        #Primary
        sample_amount = data["sample_amnt"]
        pxys = np.zeros((1,2))
    
        #calculate the actual line
        logging.info("Sampling a line")
        #start and end
        if EdgeE.BEZIER in target.data:
            for b in target.data[EdgeE.BEZIER]:
                num_points = get_distance(b[0],b[-1]) * sample_amount
                if len(b) == 3:
                    new_points = bezier1cp(*b, num_points)
                else:
                    assert(len(b) == 4)
                    new_points = bezier2cp(*b, num_points)
                pxys = np.row_stack((pxys, new_points))
        else:
            #straight line
            num_points = get_distance(se[0], se[1]) * sample_amount
            pxys = np.row_stack((pxys, createLine(*se[0], *se[1], num_points)))
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
        """ Expect an array of [x,y,radius, r,g,b,a] """
        assert(xyrcs.shape[1] == 7)
        for line in xyrcs:
            ctx.set_source_rgba(*line[3:])
            drawCircle(ctx, *line[:2], line[2])

    def getEasing(self, key, data):
        try:
            easingData = data[key]
            easing_fn = easings.lookup(easingData[0])
            if len(easingData) > 2:
                codomain = easingData[2]
            else:
                codomain = easings.CODOMAIN.FULL
            if len(easingData) > 3:
                quantize = easingData[3]
            else:
                quantize = 0

            if quantize != 0:
                easing_lambda = lambda xs: easings.quantize(easing_fn(xs, easingData[1], codomain_e=codomain), q=quantize)
            else:
                easing_lambda = lambda xs: easing_fn(xs, easingData[1], codomain_e=codomain)
            return (easing_lambda)
        except Exception as e:
            logging.warning(e)
            easing_fn = easings.lookup(easings.ENAMES[0])
            return (lambda xs: easing_fn(xs, 1, codomain_e=easings.CODOMAIN.FULL))

        
            

def defaultPost(core, secondary, target, data):
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
    colours = pcolour.repeat(len(core)).reshape((4,-1)).T
    coreMod = np.column_stack((core, radi, colours))

    secLen = len(secondary[0])
    secRadi = np.array(pradius).repeat(secLen)
    secColours = pcolour.repeat(secLen).reshape((4,-1)).T
    secMod = np.zeros((1,secLen,7))
    for a in secondary:
        b = np.column_stack((a, secRadi, secColours))
        secMod = np.row_stack((secMod, np.array([b])))
            
    return (coreMod, secMod[1:])
    
