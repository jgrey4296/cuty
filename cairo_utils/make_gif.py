import sys
from os.path import isfile,exists,join, getmtime, splitext
from os import listdir
from PIL import Image,ImageSequence
import imageio
import re
import logging as root_logger

logging = root_logger.getLogger(__name__)

class Make_Gif:
    """ A Utility class to easily create gifs from a number of images """
    
    def __init__(self, output_dir=".", gif_name="anim.gif", source_dir="images", file_format=".png", fps=12):
        self.output_dir = output_dir
        self.gif_name = gif_name
        self.source_dir = source_dir
        self.file_format = file_format
        self.fps = fps

        self.numRegex = re.compile(r'(\d+)')

    def getNum(self, s):
        """ Given a String, extract a number from it,
        or return a default """
        logging.info("Getting num of: {}".format(s))
        assert(isinstance(s, str))
        try:
            return int(self.numRegex.search(s).group(0))
        except Exception:
            return 9999999

    def run(self):
        # Get all Files
        files = [x for x in listdir(SOURCE_DIR) if isfile(join(SOURCE_DIR,x))]
        assert(bool(files))
        logging.info("Making gif of {} frames".format(len(files)))

        # Sort by the number extracted from the filename
        files.sort(key=lambda x: getNum(x))

        # Export as a Gif
        logging.info("Starting GIF writing")
        with imageio.get_writer(join(GIF_OUTPUT_DIR,GIF_NAME), mode='I') as writer:
            for filename in files:
                image = imageio.imread(join(SOURCE_DIR,filename))
                writer.append_data(image)
        logging.info("Finished GIF writing")

