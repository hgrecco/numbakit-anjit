"""
    nbkanjit.signature
    ~~~~~~~~~~~~~~~~~~

    Functions to transform annotations to signatures.

    :copyright: 2021 by nbkdeco Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import dataclasses
import inspect

import numba
from numba import types as nt
from numba.core.types import abstract

from . import exceptions

DEFAULT = object()

# Default mapping
DEFAULT_TYPE_MAPPING = {
    None: nt.void,
    int: nt.int64,
    float: nt.float64,
    bool: nt.boolean,
}


@dataclasses.dataclass
class Function:
    """Wraps a function to later get the argument, return or
    annotation as FunctionType
    """

    func: callable

    def __getattr__(self, item):
        return inspect.signature(self.func).parameters[item].annotation

    @property
    def _return(self):
        return inspect.signature(self.func).return_annotation


def map_to_numba_type(obj, mapping):
    """Map an python value to numba type.

    Parameters
    ----------
    obj : object
    mapping : dict
        A dictionary mapping python types or other values into numba types.

    Returns
    -------
    abstract.Type

    Raises
    ------
    exceptions.UnknownAnnotation
        when the object is not a valid numba type.
    """

    if is_numba_type(obj):
        return obj

    try:
        return mapping[obj]
    except KeyError:
        raise exceptions.UnknownAnnotation(obj)


def is_numba_type(obj):
    """Return True if `obj` is a numba type."""
    mro = inspect.getmro(obj.__class__)
    return abstract.Type in mro


def verify_mapping(mapping, raise_on_err=True):
    """Check if each value of the mapping is a valid numba type and
    therefore suitable to build a numba signature.

    Parameters
    ----------
    mapping : dict
        A dictionary mapping python types or other values into numba types.
    raise_on_err : bool (Default: True)
        If true, an exception is raised when an invalid value is false.
        If false, a tuple of invalid values is returned.

    Returns
    -------
    tuple of pairs
        contains the invalid keys and values from the mapping.
    Raises
    ------
    exceptions.NotNumbaSignature
        when at least one of the values is not a valid numba type.
    """

    invalid = []

    for k, v in mapping.items():
        if not is_numba_type(v):
            if raise_on_err:
                raise exceptions.NotNumbaSignature(v)
            else:
                invalid.append((k, v))

    return tuple(invalid)


def build_signature(func, mapping, on_missing_arg="raise", on_missing_ret="raise"):
    """Return a numba signature built from the annotations in the callable.

    The `mapping` dict is used to translate Python types or other values
    into numba types.

    Parameters
    ----------
    func : callable
    mapping : dict
        A dictionary mapping python types or other values into numba types.
    on_missing_arg : object (Default: "raise")
        Numba type to use when an annotation is not present for a given argument.
        By default, an exception is raised.
    on_missing_ret: object (Default: "raise")
        Numba type to use when an annotation is not present for the return value.
        By default, an exception is raised.

    Returns
    -------
    abstract.Type
        A numba type that can be used to register the signature.

    Raises
    ------
    MissingAnnotation
        When a required annotation is not found.
    """

    func_sig = inspect.signature(func)

    sig = []
    for par in func_sig.parameters.values():
        pa = par.annotation
        if pa is func_sig.empty:
            if on_missing_arg == "raise":
                raise exceptions.MissingAnnotation(par.name)
            sig.append(map_to_numba_type(on_missing_arg, mapping))
        elif isinstance(pa, Function):
            sig.append(
                nt.FunctionType(
                    build_signature(pa.func, mapping, on_missing_arg, on_missing_ret)
                )
            )
        else:
            sig.append(map_to_numba_type(pa, mapping))

    ra = func_sig.return_annotation
    if ra is func_sig.empty:
        if on_missing_ret == "raise":
            raise exceptions.MissingAnnotation("return")
        ret_type = map_to_numba_type(on_missing_ret, mapping)
    elif isinstance(ra, Function):
        ret_type = nt.FunctionType(
            build_signature(ra.func, mapping, on_missing_arg.on_missing_ret)
        )
    else:
        ret_type = map_to_numba_type(ra, mapping)

    return ret_type(*sig)


def anjit(
    func=None,
    *,
    mapping=DEFAULT,
    on_missing_arg="raise",
    on_missing_ret="raise",
    **kwargs
):
    """Annotation aware numba njit.

    Parameters
    ----------
    func : callable
        function to jit
    mapping : dict (default: DEFAULT_MAPPING)
        A dictionary mapping python types or other values into numba types.
    on_missing_arg : object (Default: "raise")
        Numba type to use when an annotation is not present for a given argument.
        By default, an exception is raised.
    on_missing_ret: object (Default: "raise")
        Numba type to use when an annotation is not present for the return value.
        By default, an exception is raised.
    **kwargs
        Extra keyword arguemnts are passed to the `numba.njit`

    Returns
    -------
    callable
        A numba jit decorated function.
    """

    if mapping is DEFAULT:
        mapping = DEFAULT_TYPE_MAPPING

    def njit_decorator(func_):
        sig = build_signature(
            func_,
            mapping=mapping,
            on_missing_arg=on_missing_arg,
            on_missing_ret=on_missing_ret,
        )
        return numba.njit(sig, **kwargs)(func_)

    if func:
        return njit_decorator(func)
    else:
        return njit_decorator
