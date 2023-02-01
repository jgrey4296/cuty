"""
Utility Module to simplify making a GIF from individual images
"""
#pylint: disable=too-many-arguments
import logging as root_logger
import re
from dataclasses import InitVar, dataclass, field
from os import listdir
from os.path import isfile, join
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    List, Mapping, Match, MutableMapping, Optional, Sequence,
                    Set, Tuple, TypeVar, Union, cast)

import imageio

logging = root_logger.getLogger(__name__)

@dataclass
class MakeGif:
    """ A Utility class to easily create gifs from a number of images """

    output_dir  : str = field(default=".")
    gif_name    : str = field(default="anim.gif")
    source_dir  : str = field(default="images")
    file_format : str = field(default=".png")
    fps         : int = field(default=12)

    num_regex = re.compile(r'(\d+)')


    def get_num(self, s):
        """ Given a String, extract a number from it,
        or return a default """
        logging.info("Getting num of: {}".format(s))
        assert(isinstance(s, str))
        #pylint: disable=broad-except
        try:
            return int(self.num_regex.search(s).group(0))
        except Exception:
            return 9999999

    def run(self):
        """ Trigger the creation of the GIF """
        # Get all Files
        files = [x for x in listdir(self.source_dir) if isfile(join(self.source_dir, x))]
        assert(bool(files))
        logging.info("Making gif of {} frames".format(len(files)))

        # Sort by the number extracted from the filename
        files.sort(key=self.get_num)

        # Export as a Gif
        logging.info("Starting GIF writing")
        with imageio.get_writer(join(self.output_dir, self.gif_name), mode='I') as writer:
            for filename in files:
                image = imageio.imread(join(self.source_dir, filename))
                writer.append_data(image)
        logging.info("Finished GIF writing")
