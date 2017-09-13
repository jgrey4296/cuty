import logging as root_logger
logging = root_logger.getLogger(__name__)


class NilNode:
    """ A Nil Node for use with the rb algorithms,
    Everything is static because the same NilNode is 
    shared for all nodes """
    canonical = False
    red = False
    parent = None
    right = None
    left = None
    arc = False
    value = None
    @staticmethod
    def __str__(self):
        return "Nil Node"

    @staticmethod
    def getMin():
        return NilNode.canonical

    @staticmethod
    def getMax():
        return NilNode.canonical

    @staticmethod
    def compare(x,d=None):
        raise Exception("Compare should not be called on NilNode")

    @staticmethod
    def compare_simple(x,d=None):
        raise Exception("Compare_simple should not be called on a NilNode")

    @staticmethod
    def isLeaf():
        return True

    @staticmethod
    def get_predecessor():
        return NilNode.canonical

    @staticmethod
    def get_successor():
        return NilNode.canonical

    @staticmethod
    def get_predecessor():
        return NilNode.canonical

    @staticmethod
    def get_successor():
        return NilNode.canonical
