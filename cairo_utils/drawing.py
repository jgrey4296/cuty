import cairo
import logging as root_logger
from .constants import TEXT, BACKGROUND, TWOPI, FRONT, FONT_SIZE
import random

logging = root_logger.getLogger(__name__)

def setup_cairo(N=5, font_size=FONT_SIZE):
    size = pow(2, N)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
    ctx = cairo.Context(surface)
    ctx.scale(size, size)
    ctx.set_font_size(font_size)
    clear_canvas(ctx)
    return (surface, ctx, size, N)


def write_to_png(surface, filename, i=None):
    if i:
        surface.write_to_png("{}_{}.png".format(filename, i))
    else:
        surface.write_to_png("{}.png".format(filename))

def drawRect(ctx, x, y, sx, sy):
    #ctx.set_source_rgba(*FRONT)
    ctx.rectangle(x, y, sx, sy)
    ctx.fill()
    
def drawCircle(ctx, x, y, r, fill=True):
    try:
        ctx.arc(x, y, r, 0, TWOPI)
    except TypeError as e:
        logging.error(x, y, r)
        raise e
    if fill:
        ctx.fill()
    else:
        ctx.stroke()

def clear_canvas(ctx, colour=BACKGROUND):
    ctx.set_source_rgba(*colour)
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()
    ctx.set_source_rgba(*FRONT)

def drawText(ctx, x, y, string, offset=False):
    if offset:
        offset = random.random() * 0.005
    else:
        offset = 0
    ctx.set_source_rgba(*TEXT)
    ctx.move_to(x+offset, y+offset)
    ctx.show_text(str(string))
