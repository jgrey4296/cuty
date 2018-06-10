import numpy as np
import IPython
from ...constants import SMALL_RADIUS, VERTEX
from ...drawing import drawRect, drawCircle, clear_canvas, drawText

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
        raise Exception("SampleSpec.primary_sample should be overriden")

    def secondary_sample(self, core, target, data):
        """ The method to override """
        raise Exception("SampleSpec.secondary_sample should be overriden")
    
    def draw_point_cloud(self, ctx, xyrcs):
        """ Expect an array of [x,y,radius, r,g,b,a] """
        assert(xyrcs.shape[1] == 7)
        for line in xyrcs:
            ctx.set_source_rgba(*line[3:])
            drawCircle(ctx, *line[:2], line[2])


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
    
