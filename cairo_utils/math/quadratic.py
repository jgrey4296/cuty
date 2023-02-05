"""
Provides a Quadratic Clss, allowing manipulation of quadratic equations
"""
##-- imports
from __future__ import annotations
import logging as root_logger
from dataclasses import InitVar, dataclass, field
from math import sqrt
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    List, Mapping, Match, MutableMapping, Optional, Sequence,
                    Set, Tuple, TypeVar, Union, cast)

import numpy as np

##-- end imports

logging = root_logger.getLogger(__name__)

@dataclass
class Quadratic:
    """ A Class to hold a quadratic equation
    (y = ax^2 + bx + c)
    and perform operations on it
    """

    a : float = field(default=0)
    b : float = field(default=0)
    c : float = field(default=0)

    def __repr__(self):
        return f"<Quadratic: {self.a} x^2 + {self.b} x + {self.x}>"

    def __call__(self, x):
        """ Shorthand to get y for a given x """
        if x is None:
            return None
        result = (self.a * pow(x, 2)) + (self.b * x) + self.c
        return result

    def intersect(self, q2):
        """ Get the x coordinates of the intersections of the two quadratics """
        assert(isinstance(q2, Quadratic))
        aprime = q2.a - self.a
        bprime = q2.b - self.b
        cprime = q2.c - self.c

        q3 = Quadratic(aprime, bprime, cprime)

        xs = q3.solve()
        xs_existing = np.array([x for x in xs if x is not None])
        xs_existing.sort()
        return xs_existing

    def discriminant(self):
        """ Get the discriminant of the quadratic """
        return pow(self.b, 2) - (4 * self.a * self.c)

    def solve(self):
        """ Solve the quadratic, returning up to two values"""
        return_value = []
        discriminant = self.discriminant()
        numerator_a  = -self.b
        denominator  = 2 * self.a

        if discriminant < 0:
            return_value = [None, None]
        elif np.allclose(discriminant, 0) or np.allclose(self.a, 0):
            logging.debug('Only one intersection')
            #using mullers method:
            twoc = -2 * self.c
            sqrt_discriminant = sqrt(discriminant)
            pos = (-self.b) + sqrt_discriminant
            neg = (-self.b) - sqrt_discriminant
            if pos != 0:
                x = twoc / pos
            elif neg != 0:
                x = twoc / neg
            else:
                logging.debug("Not even one intersection")
                x = None
            return_value = [x, None]
        else:
            z = sqrt(discriminant)
            return_value = [
                (numerator_a + z) / denominator,
                (numerator_a - z) / denominator,
            ]

        return return_value

    # def solve(self):
    #     twoA = 2 * self.a
    #     sqrtb4ac = sqrt(pow(self.b, 2) - (4 * self.a * self.c))
    #     pos = -self.b + sqrtb4ac
    #     neg = -self.b - sqrtb4ac
    #     return [neg, pos]
