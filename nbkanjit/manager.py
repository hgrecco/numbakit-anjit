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
        **kwargs
    ):

        if mapping is DEFAULT:
            mapping = signature.DEFAULT_TYPE_MAPPING

        self.mapping = mapping
        self.on_missing_arg = on_missing_arg
        self.on_missing_ret = on_missing_ret
        self.disable_jit = disable_jit
        self.kwargs = kwargs

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
