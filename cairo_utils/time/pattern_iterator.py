"""
PatternIterator automates pattern output
"""
from fractions import Fraction as f
from .utils import time_str
import logging as root_logger
logging = root_logger.getLogger(__name__)

class PatternIterator:
    """ Automates retrieval of pattern values  """

    def __init__(self, pattern, just_values=True):
        self.pattern = pattern
        self.denominator = self.pattern.denominator()
        self.position = 0
        self.just_values = just_values

    def __iter__(self):
        return self

    def __next__(self):
        time = f(self.position, self.denominator)
        self.position += 1
        return self.pattern(time, just_values=self.just_values)

    #raise StopIteration
