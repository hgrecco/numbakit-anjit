"""
    nbkanjit.exceptions
    ~~~~~~~~~~~~~~~~~~~

    numbakit-anjit exceptions

    :copyright: 2021 by nbkdeco Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


class UnknownAnnotation(ValueError):
    """The annotations is unknown and cannot be translated into a Numba type.

    Use the mapping argument to teach nbkanjit how to translate this type.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Unknown annotation, cannot translate {self.value} into a Numba type."

    def __repr__(self):
        return f"<UnknownAnnotation({self.value})>"


class NotNumbaSignature(ValueError):
    """The translated annotation is not a numba type/signature and
    therefore it cannot be used to build a numba signature.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Not a numba signature: {self.value}"

    def __repr__(self):
        return f"<NotNumbaSignature({self.value})>"


class MissingAnnotation(ValueError):
    """A required annotation is missing."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Missing annotation for {self.value}"

    def __repr__(self):
        return f"<MissingAnnotation({self.value})>"
