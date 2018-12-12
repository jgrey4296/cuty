""" Utilities for Cairo and DCEL """
import logging as root_logger
import pickle
from os.path import join, exists

logging = root_logger.getLogger(__name__)

def pickle_graph(node_dict, nx_graph, filename, directory="."):
    """ Given a graph and dictionary of nodes,
    pickle it to the specified filename """
    if not exists(directory):
        raise Exception("Directory not found: {}".format(directory))
    with open(join(directory, "{}_nxgraph.pickle".format(filename)), 'wb') as f:
        pickle.dump(nx_graph, f)
    with open(join(directory, "{}_node_dict.pickle".format(filename)), 'wb') as f:
        pickle.dump(node_dict, f)

def load_pickled_graph(filename, directory="."):
    """ Given a filename, load a pickled graph and
    its dictionary of nodes """
    if not (exists(directory) and \
             exists(join(directory, "{}_nxgraph.pickle".format(filename))) and \
             exists(join(directory, "{}_node_dict.pickle".format(filename)))):
        raise Exception("Missing file or directory for reconstruction")
    node_dict = None
    nx_graph = None
    with open(join(directory, "{}_nxgraph.pickle".format(filename)), 'rb') as f:
        nx_graph = pickle.load(f)
    with open(join(directory, "{}_node_dict.pickle".format(filename)), 'rb') as f:
        node_dict = pickle.load(f)
    return (node_dict, nx_graph)
