from ..drawing import drawRect, drawCircle, clear_canvas, drawText


class Drawable:
    """ A Basic Drawable Superclass """

    def __init__(self):
        raise Exception("Drawable Should not be instantiated")

    def draw(self, ctx):
        raise Exception("Drawble.draw is abstract. Implement it in the calling class")

    def draw_point_cloud(self, ctx, xys, rs, colours):
        assert(len(xys) == len(rs) == len(colours))
        for i in range(len(xys)):
            ctx.set_source_rgba(*colours[i])
            drawCircle(ctx, *xys[i], rs[i])







            
