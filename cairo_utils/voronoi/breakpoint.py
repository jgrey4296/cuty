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


@dataclass
class BreakWrapper:
    """ A Simple Breakpoint Wrapper """

    bp1 : Any = field()
    bp2 : Any = field()

    def __eq__(self, other):
        assert(isinstance(other, BreakWrapper))
        if self.bp1 == other.bp1 and self.bp2 == other.bp2:
            return True
        if self.bp1.value == other.bp1.value and self.bp2.value == other.bp2.value:
            return True
        return False

    def __hash__(self):
        return hash((self.bp1, self.bp2))
