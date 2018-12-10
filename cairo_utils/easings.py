""" 
Module of different easing functions
All functions take xs from -1 - 1,
and produce ys from 0 - 1
"""
import numpy as np
from enum import Enum
import IPython
from .constants import PI

CODOMAIN = Enum("Codomain of the curve", "FULL LEFT RIGHT")

def quantize(xs, r=None, q=5, bins=None):
    if r is None:
        r = [xs.min(), xs.max()]
    if bins is None:
        bins = np.linspace(r[0], r[1], q)
    inds = np.digitize(xs, bins, right=True)
    return np.array([bins[x] for x in inds])    

def interp(xs, r=None, codomain_e=CODOMAIN.FULL):
    """ Auto interpolate input to an -1 to 1 scale,
    r can override the calculated input domain of xs.min/max"""
    codomain = [-1, 1]
    if codomain_e == CODOMAIN.LEFT:
        codomain = [-1, 0]
    elif codomain_e == CODOMAIN.RIGHT:
        codomain = [0, 1]    
    if r is None:
        r = [xs.min(), xs.max()]
    return np.interp(xs, r, codomain)

def pow_abs_curve(xs, a, r=None, codomain_e=CODOMAIN.FULL):
    """ 1 - pow(abs(x),a)
    Var a changes the curve. 
    0.5: exp tri
    3.5: saturated bell """
    assert(0.5 <= a <= 3.5)
    ixs = interp(xs, r=r, codomain_e=codomain_e)
    return 1 - pow(np.abs(ixs), a)

def pow_cos_pi(xs, a, r=None, codomain_e=CODOMAIN.FULL):
    """ pow(cos(pi * x / 2) a)
    Var a changes curve
    0.5: semi circle
    3.5: gaussian window """
    assert(0.5 <= a <= 3.5)
    ixs = interp(xs, r=r, codomain_e=codomain_e)
    return pow(np.cos(PI * ixs * 0.5), a)

def pow_abs_sin_pi(xs, a, r=None, codomain_e=CODOMAIN.FULL):
    """ 1 - pow(abs(sin(pi * x / 2)), a)
    Var a changes curve
    0.5: exp_tri
    3.5: slightly saturated f
    """
    assert(0.5 <= a <= 3.5)
    ixs = interp(xs, r=r, codomain_e=codomain_e)
    return 1 - pow(np.abs(np.sin(PI * ixs * 0.5)), a)

def pow_min_cos_pi(xs, a, r=None, codomain_e=CODOMAIN.FULL):
    """ 1 - pow(min(cos(pi * x / 2), 1 - abs(x)), a)
    Var a changes curve
    0.5: leaf
    3.5: ep_tri
    """
    assert(0.5 <= a <= 3.5)
    ixs = interp(xs, r=r, codomain_e=codomain_e)
    ixs_balanced = np.column_stack((np.cos(PI * ixs * 0.5), 1 - np.abs(ixs)))
    return 1 - pow(np.min(ixs_balanced, axis=1), a)

def pow_max_abs(xs, a, r=None, codomain_e=CODOMAIN.FULL):
    """ 1 - pow(max(0, abs(x)) * 2 - 1, a)
    Var a changes curve
    0.5: exp_tri
    3.5: slightly saturated f
    """
    assert(0.5 <= a <= 3.5)
    ixs = interp(xs, r=r, codomain_e=codomain_e)
    ixs_zero = np.column_stack((np.abs(ixs) * 2 - 1, np.zeros(len(ixs))))
    return 1 - pow(np.max(ixs_zero, axis=1), a)

def sigmoid(xs,a,r=None,codomain_e=CODOMAIN.FULL):
    """ 1 / (1 + e^(-5 * x)
    Var a does nothing
    """
    ixs = interp(xs, r=r, codomain_e=codomain_e)
    return 1 / (1 + pow(np.e, -5 * ixs))

    
ELOOKUP = {
    "pow_abs_curve": pow_abs_curve,
    "pow_cos_pi": pow_cos_pi,
    "pow_abs_sin_pi": pow_abs_sin_pi,
    "pow_min_cos_pi" : pow_min_cos_pi,
    "pow_max_abs" : pow_max_abs,
    "sigmoid" : sigmoid,
}
ENAMES = list(ELOOKUP.keys())

def lookup(name):
    if isinstance(name, int):
        return ELOOKUP[ENAMES[name]]
    elif name not in ELOOKUP:
        raise Exception("Unrecognised easing name: {}".format(name))
    return ELOOKUP[name]
