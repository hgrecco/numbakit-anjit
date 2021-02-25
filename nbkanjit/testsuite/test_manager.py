import inspect

from numba import njit
from numba import types as nt

from nbkanjit import signature
from nbkanjit.manager import JitManager


def test_anjit():

    jm = JitManager()

    def _fun1(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    @jm.anjit
    def fun1a(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    @jm.anjit()
    def fun1b(x: int, y: float) -> float:
        return x + y

    d = {**signature.DEFAULT_TYPE_MAPPING, "a": nt.float64}

    @jm.anjit(mapping=d)
    def fun1c(x: int, y: float) -> float:
        return x + y

    @njit(nt.float64(nt.int64, nt.float64))
    def fun2(x, y) -> float:
        return x + y

    @jm.njit(nt.float64(nt.int64, nt.float64))
    def fun3(x, y) -> float:
        return x + y

    assert fun1a.__name__ == "fun1a"
    assert fun1a.__annotations__ == _fun1.__annotations__
    assert fun1a.__doc__ == _fun1.__doc__
    assert inspect.signature(fun1a) == inspect.signature(_fun1)
    assert fun1a.nopython_signatures
    assert fun1a.nopython_signatures == fun1b.nopython_signatures
    assert fun1a.nopython_signatures == fun1c.nopython_signatures
    assert fun1a.nopython_signatures == fun2.nopython_signatures
    assert fun1a.nopython_signatures == fun3.nopython_signatures


def test_anjit_custom_mapping():
    d = {**signature.DEFAULT_TYPE_MAPPING, "a": nt.float64}

    jm = JitManager(mapping=d)

    @jm.anjit
    def fun1(x: int, y: "a") -> float:  # noqa: F821
        """Simple doc"""
        return x + y

    @njit(nt.float64(nt.int64, nt.float64))
    def fun2(x, y):
        return x + y

    assert fun1.nopython_signatures
    assert fun1.nopython_signatures == fun2.nopython_signatures


def test_disable():
    jm = JitManager(disable_jit=True)

    @jm.anjit
    def fun1(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    assert not hasattr(fun1, "nopython_signatures")
