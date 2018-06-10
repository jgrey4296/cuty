""" 
Module of different easing functions
All functions produce xs from -1 - 1,
and ys from 0 - 1
"""
import numpy as np
from .constants import PI


def interp(xs, r=None):
    """ Auto interpolate input to an -1 to 1 scale,
    r can override the calculated input domain of xs.min/max"""
    if r is None:
        r = [xs.min(), xs.max()]
    return np.interp(xs, r, [-1,1])

def pow_abs_curve(xs, a, r=None):
    """ 1 - pow(abs(x),a)
    Var a changes the curve. 
    0.5: exp tri
    3.5: saturated bell """
    assert(0.5 <= a <= 3.5)
    ixs = interp(xs, r=r)
    return 1 - pow(np.abs(ixs), a)

def pow_cos_pi(xs, a, r=None):
    """ pow(cos(pi * x / 2) a)
    Var a changes curve
    0.5: semi circle
    3.5: gaussian window """
    assert(0.5 <= a <= 3.5)
    ixs = interp(xs, r=r)
    return pow(np.cos(PI * isxs * 0.5), a)

def pow_abs_sin_pi(xs, a, r=None):
    """ 1 - pow(abs(sin(pi * x / 2)), a)
    Var a changes curve
    0.5: exp_tri
    3.5: slightly saturated f
    """
    assert(0.5 <= a <= 3.5)
    ixs = interp(xs, r=r)
    return 1 - pow(np.abs(np.sin(pi * ixs * 0.5), a))

def pow_min_cos_pi(xs, a, r=None):
    """ 1 - pow(min(cos(pi * x / 2), 1 - abs(x)), a)
    Var a changes curve
    0.5: leaf
    3.5: ep_tri
    """
    assert(0.5 <= a <= 3.5)
    ixs = interp(xs, r=r)
    return 1 - pow(np.min(np.cos(PI * xs * 0.5)), 1 - np.abs(xs), a)

def pow_max_abs(xs, a, r=None):
    """ 1 - pow(max(0, abs(x)) * 2 - 1, a)
    Var a changes curve
    0.5: exp_tri
    3.5: slightly saturated f
    """
    assert(0.5 <= a <= 3.5)
    ixs = interp(xs, r=r)
    return 1 - pow(np.max(0, np.abs(xs)) * 2 - 1, a)
