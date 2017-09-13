""" Utilities for Cairo and DCEL """
import math
from math import sin, cos, atan2
from random import choice
import logging
import sys
import random
import numpy as np
from numpy import pi
import numpy.random
from scipy.interpolate import splprep, splev
import IPython
import pickle
from os.path import join,isfile,exists,isdir
from os import listdir


DRAW_TEXT = True

def pickle_graph(node_dict,nx_graph,filename,directory="."):
    if not exists(directory):
        raise Exception("Directory not found: {}".format(directory))
    with open(join(directory,"{}_nxgraph.pickle".format(filename)),'wb') as f:
        pickle.dump(nx_graph,f)
    with open(join(directory,"{}_node_dict.pickle".format(filename)),'wb') as f:
        pickle.dump(node_dict,f)

def load_pickled_graph(filename,directory="."):
    if not ( exists(directory) and \
             exists(join(directory,"{}_nxgraph.pickle".format(filename))) and \
             exists(join(directory,"{}_node_dict.pickle".format(filename))) ):
        raise Exception("Missing file or directory for reconstruction")
    node_dict = None
    nx_graph = None
    with open(join(directory,"{}_nxgraph.pickle".format(filename)),'rb') as f:
        nx_graph = pickle.load(f)
    with open(join(directory,"{}_node_dict.pickle".format(filename)),'rb') as f:
        node_dict = pickle.load(f)
    return (node_dict,nx_graph)


