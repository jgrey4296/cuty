""" Provides the basic Drawing Superclass """
##-- imports
from __future__ import annotations
import logging as root_logger
from dataclasses import InitVar, dataclass, field
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    List, Mapping, Match, MutableMapping, Optional, Sequence,
                    Set, Tuple, TypeVar, Union, cast)

from ..drawing import draw_circle

##-- end imports

logging = root_logger.getLogger(__name__)

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
