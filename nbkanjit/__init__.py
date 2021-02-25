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

try:
    from importlib.metadata import version
except ImportError:
    # Backport for Python < 3.8
    from importlib_metadata import version

try:  # pragma: no cover
    __version__ = version("numbakit-anjit")
except Exception:  # pragma: no cover
    # we seem to have a local copy not installed without setuptools
    # so the reported version will be unknown
    __version__ = "unknown"

del version

__all__ = [
    "UnknownAnnotation",
    "NotNumbaSignature",
    "MissingAnnotation",
    "JitManager",
    "anjit",
    "Function",
]
