#!/usr/bin/env python3
"""

"""
##-- imports
from __future__ import annotations

import types
import abc
import datetime
import enum
import functools as ftz
import itertools as itz
import logging as logmod
import pathlib as pl
import re
import time
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Final, Generic,
                    Iterable, Iterator, Mapping, Match, MutableMapping,
                    Protocol, Sequence, Tuple, TypeAlias, TypeGuard, TypeVar,
                    cast, final, overload, runtime_checkable)
from uuid import UUID, uuid1
from weakref import ref

##-- end imports

##-- logging
logging = logmod.getLogger(__name__)
##-- end logging

import cairo
import cauto_utils as cu
import cairo_utils.constants as constants
from cairo_utils.struct.colour import Colour

@dataclass
class DrawSettings:
    """
    Utility a Cairo surface and context
    n : the pow2 size of the surface
    font_size
    scale : True for coords of -1 to 1
    cartesian : True for (0,0) being in the bottom left, instead of top right
    background : The background colour to initialize to
    """
    size       : int    = field(default=5)
    font_size  : int    = field(default=constants.FONT_SIZE)
    scale      : bool   = field(default=True)
    cartesian  : bool   = field(default=False)
    background : Colour = field(default_factory=Colour)

    @property
    def full_size(self):
        return pow(2, self.size)

class DrawMixin:

    settings : DrawSettings      = None
    surface : cairo.ImageSurface = None
    ctx     : cairo.Context      = None

    def setup_cairo(self, settings:DrawSettings):
        size          = settings.full_size
        self.settings = settings
        self.surface  = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
        self.ctx      = cairo.Context(surface)

        if settings.cartesian:
            self.ctx.scale(1, -1)
            self.ctx.translate(0, -size)
        if settings.scale:
            self.ctx.scale(size, size)
        self.ctx.set_font_size(font_size)
        self.clear_canvas(colour=settings.background)
        return (surface, ctx, size, n)

    def write_to_png(self, filename, i=None):
        """ Write the given surface to a png, with optional numeric postfix
        surface : The surface to write
        filename : Does not need file type postfix
        i : optional numeric
        """
        logging.info("Drawing To File")
        if i:
            self.surface.write_to_png("{}_{}.png".format(filename, i))
        else:
            self.surface.write_to_png("{}.png".format(filename))

    def draw_rect(self, xyxys:np.array, fill=True):
        """ Draw simple rectangles.
        Takes the context and a (n,4) array
        """
        #ctx.set_source_rgba(*FRONT)
        match fill:
            case False:
                draw_fn = lambda c: c.stroke()
            case True:
                draw_fn = lambda c: c.fill()

        match xyxys.shape[0]:
            case 7:
                set_colour = lambda c, x: c.set_source_rgba(*x[-4:])
            case _:
                set_colour = lambda c, x: None

        rect_fn = lambda xyr: [*xyr, xyr[-1]]

        for pairs in xyxys:
            set_colour(self.ctx, pairs)
            self.ctx.rectangle(*rect_fn(pairs[:3]))
            draw_fn(self.ctx)

    def draw_circle(self, xyrs, fill=True):
        """ Draw simple circles
        Takes context,
        """
        match fill:
            case False:
                draw_fn = lambda c: c.stroke()
            case True:
                draw_fn = lambda c: c.fill()

        match xyxys.shape[0]:
            case cu.config.SAMPLE_DATA_LEN:
                set_colour = lambda c, x: c.set_source_rgba(*x[-COLOUR_SIZE:])
            case _:
                set_colour = lambda c, x: None

        for data in xyrs:
            set_colour(self.ctx, data)
            self.ctx.arc(*data[:3], 0, constants.TWOPI)
            draw_fn(self.ctx)

    def draw_text(self, xy:np.array, text:str):
        """ Utility to simplify drawing text
        Takes context, position, text
        """
        logging.debug("Drawing text: %s, %s", text, xy)
        self.ctx.save()
        self.ctx.move_to(*xy)
        self.ctx.scale(1, -1)
        self.ctx.show_text(str(text))
        self.ctx.scale(1, -1)
        self.ctx.restore()

    def clear_canvas(self, colour=constants.BACKGROUND, bbox=None):
        """ Clear a rectangle of a context using particular colour
        colour : The colour to clear to
        bbox : The area to clear, defaults to (0,0,1,1)
        """
        self.ctx.set_source_rgba(*colour)
        if bbox is None:
            self.ctx.rectangle(0, 0, 1, 1)
        else:
            self.ctx.rectangle(*bbox)
        self.ctx.fill()
        self.ctx.set_source_rgba(*cu.data.FRONT)
