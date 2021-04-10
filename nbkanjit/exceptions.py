"""
    nbkanjit.exceptions
    ~~~~~~~~~~~~~~~~~~~

    numbakit-anjit exceptions

    :copyright: 2021 by nbkdeco Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


class AnjitException(Exception):
    """Base class from all exceptions in exceptions."""

    _extra_info = None

    def append_info(self, s):
        if self._extra_info is None:
            self._extra_info = []

        self._extra_info.append(s)

    @property
    def extra_info(self):
        if self._extra_info:
            return "\n" + "\n".join(self._extra_info)
        return ""


class UnknownAnnotation(ValueError, AnjitException):
    """The annotations is unknown and cannot be translated into a Numba type.

    Use the mapping argument to teach nbkanjit how to translate this type.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Unknown annotation, cannot translate {self.value} into a Numba type.{self.extra_info}"

    def __repr__(self):
        return f"<UnknownAnnotation({self.value})>"


class NotNumbaSignature(ValueError, AnjitException):
    """The translated annotation is not a numba type/signature and
    therefore it cannot be used to build a numba signature.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Not a numba signature: {self.value}.{self.extra_info}"

    def __repr__(self):
        return f"<NotNumbaSignature({self.value})>"


class MissingAnnotation(ValueError, AnjitException):
    """A required annotation is missing."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Missing annotation for {self.value}.{self.extra_info}"

    def __repr__(self):
        return f"<MissingAnnotation({self.value})>"
