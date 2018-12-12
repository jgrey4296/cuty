"""
Utility Module to simplify making a GIF from individual images
"""
#pylint: disable=too-many-arguments
import logging as root_logger
from os.path import isfile, join
from os import listdir
import re
import imageio



logging = root_logger.getLogger(__name__)

class MakeGif:
    """ A Utility class to easily create gifs from a number of images """

    def __init__(self, output_dir=".",
                 gif_name="anim.gif",
                 source_dir="images",
                 file_format=".png", fps=12):
        self.output_dir = output_dir
        self.gif_name = gif_name
        self.source_dir = source_dir
        self.file_format = file_format
        self.fps = fps

        self.num_regex = re.compile(r'(\d+)')

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
