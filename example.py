#!/usr/bin/env python3
##-- imports
from __future__ import annotations

import sys
import time
import math
import cairo
import logging
import numpy as np
# import cairo_utils as utils
# from cairo_utils.dcel.constants import VertE, EdgeE, FaceE
import argparse
# from noise import pnoise2, snoise2
import pathlib as pl
##-- end imports

##-- logging
LOGLEVEL = logging.DEBUG
LOG_FILE_NAME = "log.{}".format(pl.Path(__file__).stem)
logging.basicConfig(filename=LOG_FILE_NAME,level=LOGLEVEL,filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
##-- end logging

##-- constants
N             = 12
SIZE          = pow(2,N)
SCALER        = 1 / SIZE
TIME          = 100
imgPath       = pl.Path()
imgName       = "initialTest"
dcel_filename = "theDCEL.pickle"
currentTime   = time.gmtime()
FONT_SIZE     = 0.03
SCALE         = False
##-- end constants

##-- argparse
parser = argparse.ArgumentParser("")
parser.add_argument('-l', "--loaddcel", action="store_true")
parser.add_argument('-s', '--static', action="store_true")
parser.add_argument('-d', '--dontdraw',action="store_true")
parser.add_argument('--drawsteps', action="store_true")
parser.add_argument('-n', '--numpoints',type=int, default=N)
parser.add_argument('-t', '--timesteps', type=int, default=TIME)
##-- end argparse

import cairo_utils
from cairo_utils.mixins.drawing_mixin import DrawMixin, DrawSettings

draw_settings = DrawSettings(N, background=[0,1,0,1])

class TestCairo(DrawMixin):
    pass



def main():
    args = parser.parse_args()

    #format the name of the image to be saved thusly:
    save_path = (imgPath / "{}_{}_{}-{}_{}-{}".format(imgName,
                                                      currentTime.tm_min,
                                                      currentTime.tm_hour,
                                                      currentTime.tm_mday,
                                                      currentTime.tm_mon,
                                                      currentTime.tm_year)
                 ).with_suffix(".png")



    draw_obj = TestCairo()
    draw_obj.setup_cairo(draw_settings)
    draw_obj.draw_rect(np.array([[0.0, 0.0, 1.0, 1, 1, 0, 1, 1],
                                 [0.0, 0.0, 0.5, 0.5, 0, 1, 0, 1]
                                ]))

    logging.info("Drawing to: {}".format(save_path))
    draw_obj.write_to_png(save_path)


##-- ifmain
if __name__ == "__main__":
    main()
##-- end ifmain
