"""
    nbkanjit
    ~~~~~~~~

    Annotation aware numba njit.

    :copyright: 2021 by nbkdeco Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from .exceptions import MissingAnnotation, NotNumbaSignature, UnknownAnnotation
from .manager import JitManager
from .signature import Function, anjit

__all__ = [
    "UnknownAnnotation",
    "NotNumbaSignature",
    "MissingAnnotation",
    "JitManager",
    "anjit",
    "Function",
]
