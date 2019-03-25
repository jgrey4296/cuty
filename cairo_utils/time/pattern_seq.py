"""
A Subclass of Pattern to chain sequences
"""
from math import floor
from .pattern import Pattern

class PatternSeq(Pattern):

    def __call__(self, count, just_values=False):
        """ Query the Pattern for a given time """
        f_count = floor(count)
        mod_f = f_count % len(self.components)
        return self.components[mod_f](count, just_values)
