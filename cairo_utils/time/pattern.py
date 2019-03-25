"""
A Pattern collects events together and cycles them
"""

from fractions import Fraction, gcd
from functools import reduce
from math import floor
import logging as root_logger
from .utils import TIME_T
import IPython

logging = root_logger.getLogger(__name__)

class Pattern:
    """ A Collection of Events """

    @staticmethod
    def lcm(a,b):
        """ Get the lowest common multiple for two fractions """
        x = abs(a)
        gcd_result = gcd(a,b)
        x_prime = x / gcd_result
        y = abs(b)
        result = x_prime * y
        return result

    def __init__(self, a, vals=None):
        if vals is None:
            vals = []
        self.arc = a.copy()
        # components :: [ Event || Pattern ]
        self.components = sorted(vals, key=lambda x: x.key())
        self.time_type = TIME_T.CLOCK

    def __call__(self, count, just_values=False):
        """ Query the Pattern for a given time """
        pattern_range = self.arc.size()
        f_count = floor(count)
        position = count - (f_count * (f_count >= pattern_range))
        scaled_position = position / pattern_range
        results = []
        for x in self.components:
            results += x(scaled_position)

        if just_values:
            results = [x.values for x in results]

        return results

    def key(self):
        """ Key the Pattern by its start time, for sorting """
        return self.arc.start

    def __contains__(self, other):
        """ Test whether a given object or time is within this patterns bounds """
        return other in self.arc

    def base(self):
        """ Get all used fractions within this arc, scaled appropriately by offset
        and pattern size """
        counts = set()
        pattern_range = self.arc.size()
        for x in self.components:
            counts.update([a * pattern_range for a in x.base()])
        return counts

    def denominator(self):
        #TODO: use https://stackoverflow.com/questions/49981286/
        base_count = reduce(gcd, self.base(), 2).denominator
        return base_count

    def __repr__(self):
        base_count = self.denominator()
        collection = []
        for y in range(base_count):
            q = self(Fraction(y, base_count), True)
            collection.append(q)

        most_simultaneous = max([len(x) for x in collection])
        rows = [[] for x in range(most_simultaneous)]
        output = "\n|" + ("-" * base_count) + "|\n"
        for x in collection:
            len_x = len(x)
            for j,r in enumerate(rows):
                if j < len_x:
                    r.append(x[j])
                else:
                    r.append('*')

        for row in rows:
            output += "|{}|\n".format("".join(row))

        return output

    def __str__(self):
        """ Print in the same format the parser reads """
        return repr(self)
