import logging as root_logger
from .constants import TEXT, BACKGROUND, TWOPI, FRONT
import random

logging = root_logger.getLogger(__name__)

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

def clear_canvas(ctx):
    ctx.set_source_rgba(*BACKGROUND)
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()
    ctx.set_source_rgba(*FRONT)

def drawText(ctx, x, y, string):
    offset = random.random() * 0.005
    ctx.set_source_rgba(*TEXT)
    ctx.move_to(x+offset, y+offset)
    ctx.show_text(str(string))
