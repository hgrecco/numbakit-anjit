"""
    nbkanjit.testsuite
    ~~~~~~~~~~~~~~~~~~

    Annotation aware numba njit.

    :copyright: 2021 by nbkode Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


def run():
    """Run all tests."""

    try:
        import pytest
    except ImportError:
        print("pytest not installed. Install it\n    pip install pytest")
        raise

    return pytest.main()
