"""
A Function to create a heightmap, quantized,
uses the red channel
"""
##-- imports
from __future__ import annotations
import numpy as np
from scipy.ndimage import generic_filter
from functools import partial
from noise import snoise2

import logging as root_logger
##-- end imports

logging = root_logger.getLogger(__name__)

baseheight = np.array([0.3, 0, 0, 0])

class HeightMap:

    @staticmethod
    def build_with_edges(size, minheight, layers, oct=3, repeatx=None, repeaty=None, base=0.0):
        """ Generates a heightmap, and the edges """
        hm    = HeightMap.build(size, minheight, oct=oct, repeatx=repeatx, repeaty=repeaty, base=base)
        qhm   = quantize(hm, layers)
        edges = HeightMap.detect_edges(qhm, layers)
        return (hm, qhm, edges)

    @staticmethod
    def build(size, minheight, oct=3, repeatx=None, repeaty=None, base=0.0):
        """ Generate a height map """
        matrix = np.zeros((size,size))
        SCALER = 1 / size

        for x in range(0, size):
            for y in range(0, size):
                xs = x * SCALER
                ys = y * SCALER
                value = snoise2(xs, ys, octaves=oct, repeatx=repeatx,
                                repeaty=repeaty, base=base)
                matrix[x,y] = value

        mmin = matrix.min()
        if mmin < minheight:
            matrix += minheight - mmin
            matrix /= matrix.max()

        return matrix


    @staticmethod
    def detect_edges(matrix, layers):
        """ Given a quantized heightmap and the number of steps,
        find the edges using convolution """
        logging.info("Detecting Edges")
        edges = np.zeros(matrix.shape)
        scaling = 100
        scaling_recip = 1 / scaling
        quantize_base = int(scaling / layers)
        scaled_matrix = matrix * scaling

        #For each level of quantization:
        for index in range(layers, 1, -1):
            layer_val = index * quantize_base
            #merge with layers above and filter
            use_matrix = scaled_matrix * (scaled_matrix >= layer_val)
            filter_mat = generic_filter(use_matrix, partial(HeightMap.filter_func, layer_val), size=2)
            #aggregate:
            edges += filter_mat

        return edges

    @staticmethod
    def filter_func(max_val, arr):
        """ A partial filter function to detect edges """
        return 0 in arr and any(arr >= max_val)
