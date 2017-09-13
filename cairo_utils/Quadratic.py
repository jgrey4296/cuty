from math import sqrt, trunc
import numpy as np
import logging
import IPython

class Quadratic(object):

    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self,x):
        if x is None:
            return None
        result = (self.a * pow(x,2)) + (self.b * x) + self.c
        return -result

    def intersect(self,q2):
        """ Get the x coordinates of the intersections of the two quadratics """
        aprime = q2.a - self.a
        bprime = q2.b - self.b
        cprime = q2.c - self.c
        
        q3 = Quadratic(aprime,bprime,cprime)
        
        xs = q3.solve()
        xs_existing = np.array([x for x in xs if x is not None])
        return xs_existing

    def discriminant(self):
        return pow(self.b,2) - (4 * self.a * self.c)

    def solve(self):
        returnVal = None
        D = self.discriminant()
        numerator_a = -self.b
        denominator = 2 * self.a
        if D < 0:
            returnVal = [None,None]
        elif np.allclose(D,0) or np.allclose(self.a,0):
            logging.debug('Only one intersection')
            #using mullers method:
            twoc = -2 * self.c
            sqrtb4ac = sqrt(pow(self.b,2) - (4 * self.a * self.c))
            pos = self.b + sqrtb4ac
            neg = self.b - sqrtb4ac
            if pos != 0:
                x = twoc / pos
            elif neg != 0:
                x = twoc / neg
            else:
                logging.debug("Not even one intersection")
                x = None
            returnVal = [x,None]
        else:
            z = sqrt(D)
            returnVal = [
                (numerator_a + z) / denominator,
                (numerator_a - z) / denominator,
            ]
        return returnVal
        
