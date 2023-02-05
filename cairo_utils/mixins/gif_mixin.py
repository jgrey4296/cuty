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
import imageio

import cairo_utils as cu

img_types = cu.config.on_fail([".jpg"], list).tool.cairo.exts()

@dataclass
class GifData:
    """ A Utility class to easily create gifs from a number of images """

    output_dir  : str = field(default=".")
    gif_name    : str = field(default="anim.gif")
    source_dir  : str = field(default="images")
    file_format : str = field(default=".png")
    fps         : int = field(default=12)

    num_regex = re.compile(r'(\d+)')

class MakeGif:

    def get_num(self, text) -> int:
        """ Given a String, extract a number from it,
        or return a default """
        logging.info("Getting num of: %s", text)
        assert(isinstance(text, str))
        try:
            return int(self.num_regex.search(s).group(0))
        except ValueException:
            return 9999999

    def make_gif(self, fpath:pl.Path):
        """ Trigger the creation of the GIF """
        # Get all Files
        frames = [x for x in fpath.iterdir() if x.suffix in img_types]
        assert(bool(frames))
        logging.info("Making gif of %s frames", len(frames))

        # Export as a Gif
        logging.info("Starting GIF writing")
        with imageio.get_writer(self.output, mode='I') as writer:
            # Sort by the number extracted from the filename
            for pframe in sorted(files, key=self.get_num):
                image = imageio.imread(pframe))
                writer.append_data(image)

        logging.info("Finished GIF writing")
