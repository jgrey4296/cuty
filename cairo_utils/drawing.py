"""
Drawing: provides basic setup and drawing utilities for cairo:
"""

import logging as root_logger
import cairo
from .constants import BACKGROUND, TWOPI, FRONT, FONT_SIZE, SAMPLE_DATA_LEN, COLOUR_SIZE

logging = root_logger.getLogger(__name__)

def setup_cairo(n=5,
                font_size=FONT_SIZE,
                scale=True,
                cartesian=False,
                background=BACKGROUND):
    """
    Utility a Cairo surface and context
    n : the pow2 size of the surface
    font_size
    scale : True for coords of -1 to 1
    cartesian : True for (0,0) being in the bottom left, instead of top right
    background : The background colour to initialize to
    """
    size = pow(2, n)
    #pylint: disable=no-member
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
    ctx = cairo.Context(surface)
    #pylint: enable=no-member
    if cartesian:
        ctx.scale(1, -1)
        ctx.translate(0, -size)
    if scale:
        ctx.scale(size, size)
    ctx.set_font_size(font_size)
    clear_canvas(ctx, colour=background)
    return (surface, ctx, size, n)

def write_to_png(surface, filename, i=None):
    """ Write the given surface to a png, with optional numeric postfix
    surface : The surface to write
    filename : Does not need file type postfix
    i : optional numeric
    """
    logging.info("Drawing To File")
    if i:
        surface.write_to_png("{}_{}.png".format(filename, i))
    else:
        surface.write_to_png("{}.png".format(filename))

def draw_rect(ctx, xyxys, fill=True):
    """ Draw simple rectangles.
    Takes the context and a (n,4) array
    """
    #ctx.set_source_rgba(*FRONT)
    f = lambda c: c.stroke()
    if fill:
        f = lambda c: c.fill()
    c = lambda c, x: None
    if len(xyxys[0]) == 7:
        c = lambda c, x: c.set_source_rgba(*x[-4:])
    q = lambda xyr: [*xyr, xyr[-1]]

    for a in xyxys:
        c(ctx, a)
        ctx.rectangle(*q(a[:3]))
        f(ctx)

def draw_circle(ctx, xyrs, fill=True):
    """ Draw simple circles
    Takes context,
    """
    f = lambda c: c.stroke()
    if fill:
        f = lambda c: c.fill()
    col = lambda c, x: None
    if len(xyrs[0]) == SAMPLE_DATA_LEN:
        col = lambda c, x: c.set_source_rgba(*x[-COLOUR_SIZE:])

    for a in xyrs:
        col(ctx, a)
        ctx.arc(*a[:3], 0, TWOPI)
        f(ctx)

def draw_text(ctx, xy, text):
    """ Utility to simplify drawing text
    Takes context, position, text
    """
    logging.debug("Drawing text: {}, {}".format(text, xy))
    ctx.save()
    ctx.move_to(*xy)
    ctx.scale(1, -1)
    ctx.show_text(str(text))
    ctx.scale(1, -1)
    ctx.restore()

def clear_canvas(ctx, colour=BACKGROUND, bbox=None):
    """ Clear a rectangle of a context using particular colour
    colour : The colour to clear to
    bbox : The area to clear, defaults to (0,0,1,1)
    """
    ctx.set_source_rgba(*colour)
    if bbox is None:
        ctx.rectangle(0, 0, 1, 1)
    else:
        ctx.rectangle(*bbox)
    ctx.fill()
    ctx.set_source_rgba(*FRONT)
