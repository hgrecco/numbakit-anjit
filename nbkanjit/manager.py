"""
    nbkanjit.manager
    ~~~~~~~~~~~~~~~~

    Defines the JitManager class to make defaults easier.

    :copyright: 2021 by nbkdeco Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import numba

from . import signature
from .signature import DEFAULT


class JitManager:
    """A convenience class to simplify jitting.

    Parameters
    ----------
    mapping : dict
        A dictionary mapping python types or other values into numba types.
    on_missing_arg : object (Default: "raise")
        Numba type to use when an annotation is not present for a given argument.
        By default, an exception is raised.
    on_missing_ret: object (Default: "raise")
        Numba type to use when an annotation is not present for the return value.
        By default, an exception is raised.
    disable_jit: bool (Default: False)
        If true, `anjit` and `njit` becomes noop.
    **kwargs
        Extra arguments which are passed to `anjit` and `njit`.
    """

    def __init__(
        self,
        *,
        mapping=DEFAULT,
        on_missing_arg="raise",
        on_missing_ret="raise",
        disable_jit=False,
        **kwargs,
    ):

        if mapping is DEFAULT:
            mapping = signature.DEFAULT_TYPE_MAPPING

        self.mapping = mapping
        self.on_missing_arg = on_missing_arg
        self.on_missing_ret = on_missing_ret
        self.disable_jit = disable_jit
        self.kwargs = kwargs

        self._signatures = {}

    def njit(self, signature_or_function, *args, **kwargs):
        if self.disable_jit:
            return signature_or_function
        return numba.njit(signature_or_function, *args, **{**self.kwargs, **kwargs})

    def anjit(self, func=None, **kwargs):
        if self.disable_jit:
            return func
        kwargs = {
            "mapping": self.mapping,
            "on_missing_arg": self.on_missing_arg,
            "on_missing_ret": self.on_missing_ret,
            **self.kwargs,
            **kwargs,
        }
        return signature.anjit(func, **kwargs)

    def register(self, name_or_func, func=None, *, raise_on_duplicate=True):
        if func is None:
            if isinstance(name_or_func, str):

                def _deco(func):
                    return self.register(
                        name_or_func, func, raise_on_duplicate=raise_on_duplicate
                    )

                return _deco

            elif not callable(name_or_func):
                raise ValueError(
                    "If only a single argument is used, it must be a callable."
                )

            else:
                name_or_func, func = name_or_func.__name__, name_or_func

        if raise_on_duplicate and name_or_func in self._signatures:
            raise ValueError(f"{name_or_func} already registered.")

        if signature.is_numba_signature(func):
            self._signatures[name_or_func] = func
        else:
            b = signature.Builder(
                self.mapping, self.on_missing_arg, self.on_missing_ret
            )
            sig = b.build_signature(func)
            self._signatures[name_or_func] = sig

    def njit_from_name(self, name_or_func=None, **kwargs):
        if name_or_func is None:

            def _deco(func):
                return self.njit_from_name(func, **kwargs)

            return _deco

        if callable(name_or_func):
            name_or_func, func = name_or_func.__name__, name_or_func
        else:
            func = None

        def njit_decorator(func_):
            sig = self._signatures[name_or_func]
            return numba.njit(sig, **kwargs)(func_)

        if func:
            if isinstance(func, staticmethod):
                return staticmethod(njit_decorator(func.__func__))
            return njit_decorator(func)
        else:
            return njit_decorator
