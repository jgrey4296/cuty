#!/usr/bin/env python3
"""

"""
##-- imports
from __future__ import annotations

import types
import abc
import datetime
import enum
import functools as ftz
import itertools as itz
import logging as logmod
import pathlib as pl
import re
import time
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Final, Generic,
                    Iterable, Iterator, Mapping, Match, MutableMapping,
                    Protocol, Sequence, Tuple, TypeAlias, TypeGuard, TypeVar,
                    cast, final, overload, runtime_checkable)
from uuid import UUID, uuid1
from weakref import ref

##-- end imports

##-- logging
logging = logmod.getLogger(__name__)
##-- end logging

import numpy as np

def quantize(matrix:np.ndarray, layers:int):
    """ Given a basic heightmap, quantize it """
    logging.info("Quantizing")
    #pad the size a bit
    qmatrix       = np.zeros(matrix.shape)
    scaling       = 100
    scaling_recip = 1 / scaling
    quantize_base = int(scaling / layers)
    scaled_matrix = matrix * scaling

    for l in range(1,layers):
        quantize_value =  quantize_base * l
        qmatrix        += scaled_matrix > quantize_value

    qmatrix *= quantize_base
    qmatrix *= scaling_recip
    return qmatrix
