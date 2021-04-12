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

    _func: callable

    @property
    def name(self):
        return self._func.__name__

    def __getattr__(self, item):
        if item == "__name":
            return inspect.signature(self._func).parameters["_name"].annotation

        elif item == "_name":
            return inspect.signature(self._func).parameters["name"].annotation

        return inspect.signature(self._func).parameters[item].annotation

    @property
    def _return(self):
        return inspect.signature(self._func).return_annotation

    def __hash__(self):
        return hash(self._func)


Return = lambda func: Function(func)._return


def is_numba_type(obj):
    """Return True if `obj` is a numba type."""
    mro = inspect.getmro(obj.__class__)
    return abstract.Type in mro


def is_numba_signature(obj):
    """Return True if `obj` is a numba type."""
    mro = inspect.getmro(obj.__class__)
    return numba.core.typing.templates.Signature in mro


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


class Builder:
    def __init__(self, mapping, on_missing_arg="raise", on_missing_ret="raise"):
        self.mapping = mapping
        self.on_missing_arg = on_missing_arg
        self.on_missing_ret = on_missing_ret

    def map_to_numba_type(self, obj):
        """Map an python type to numba type.

        Parameters
        ----------
        obj : object

        Returns
        -------
        abstract.Type

        Raises
        ------
        exceptions.UnknownAnnotation
            when the object is not a valid numba type.
        """

        # TODO: make it work for all containers
        if isinstance(obj, nt.Tuple):
            return nt.Tuple(tuple(self.map_to_numba_type(o) for o in obj))

        if isinstance(obj, Function):
            return nt.FunctionType(self.build_signature(obj._func))

        if is_numba_type(obj):
            return obj

        try:
            return self.mapping[obj]
        except KeyError:
            raise exceptions.UnknownAnnotation(obj)

    def convert_annotation(self, name, annotation, missing_value, empty):
        """Convert Python annotation to numba annotation.

        Parameters
        ----------
        name : str
            Name of the argument.
        annotation
        missing_value
            Value to be used when
        empty
            Value that represents an empty annotation
            (TODO is this necessary or is always the same?)

        Returns
        -------
        abstract.Type

        Raises
        ------
        exceptions.MissingAnnotation
            when the annotation is missing a not default was provided.
        exceptions.UnknownAnnotation
            when the object is not a valid numba type.
        """
        if annotation is empty:
            if missing_value == "raise":
                raise exceptions.MissingAnnotation(name)

            return self.map_to_numba_type(missing_value)
        else:
            return self.map_to_numba_type(annotation)

    def build_signature(self, func):
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

        if isinstance(func, staticmethod):
            func = func.__func__
        func_sig = inspect.signature(func)

        pars = []
        for par in func_sig.parameters.values():
            try:
                p = self.convert_annotation(
                    par.name, par.annotation, self.on_missing_arg, func_sig.empty
                )
            except exceptions.MissingAnnotation as ex:
                ex.append_info(f"_func: {func}")
                raise ex
            except exceptions.AnjitException as ex:
                ex.append_info(
                    f"_func: {func}. Argument: {par.name}, Annotation: {par.annotation}"
                )
                raise ex

            pars.append(p)

        try:
            ret = self.convert_annotation(
                "return",
                func_sig.return_annotation,
                self.on_missing_ret,
                func_sig.empty,
            )
        except exceptions.MissingAnnotation as ex:
            ex.append_info(f"_func: {func}")
            raise ex
        except exceptions.AnjitException as ex:
            ex.append_info(
                f"_func: {func}. Return annotation: {func_sig.return_annotation}"
            )
            raise ex

        return ret(*pars)


def anjit(
    func=None,
    *,
    mapping=DEFAULT,
    on_missing_arg="raise",
    on_missing_ret="raise",
    **kwargs,
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
        Extra keyword arguments are passed to the `numba.njit`

    Returns
    -------
    callable
        A numba jit decorated function.
    """

    if mapping is DEFAULT:
        mapping = DEFAULT_TYPE_MAPPING

    def njit_decorator(func_):
        b = Builder(mapping, on_missing_arg, on_missing_ret)
        sig = b.build_signature(func_)
        return numba.njit(sig, **kwargs)(func_)

    if func:
        if isinstance(func, staticmethod):
            return staticmethod(njit_decorator(func.__func__))
        return njit_decorator(func)
    else:
        return njit_decorator


def build_signature(func, mapping=DEFAULT, **kwargs):
    """Short for Builder(mapping, **kwargs).signature(func)

    Parameters
    ----------
    func : callable
        Function to build a numba signture
    mapping : dict (default: DEFAULT_MAPPING)
        A dictionary mapping python types or other values into numba types.
    **kwargs
        Extra keyword arguments are passed to Builder.

    Returns
    -------
    numba signature
    """
    b = Builder(mapping=mapping, **kwargs)
    return b.build_signature(func)
