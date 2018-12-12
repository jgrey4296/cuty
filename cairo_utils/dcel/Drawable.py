""" Provides the basic Drawing Superclass """
#pylint: disable=no-self-use
from ..drawing import draw_circle


class Drawable:
    """ A Basic Drawable Superclass """

    def __init__(self):
        raise Exception("Drawable Should not be instantiated")

    def draw(self, ctx):
        """ Abstract method that Drawbles implement  """
        raise Exception("Drawble.draw is abstract. Implement it in the calling class")

    def draw_point_cloud(self, ctx, xys, rs, colours):
        """ Draw a collection of points """
        assert(len(xys) == len(rs) == len(colours))
        for (i, a) in enumerate(xys):
            ctx.set_source_rgba(*colours[i])
            draw_circle(ctx, *a, rs[i])
