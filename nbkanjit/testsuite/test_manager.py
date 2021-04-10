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


def test_register():
    def fun1(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    jm = JitManager()

    sig = nt.float64(nt.int64, nt.float64)
    assert signature.is_numba_signature(sig)

    jm.register("simple", sig)
    assert jm._signatures["simple"] == sig

    jm.register(fun1)
    assert jm._signatures["fun1"] == sig

    jm.register("fun1b", fun1)
    assert jm._signatures["fun1b"] == sig

    @jm.register
    def fun2(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    assert jm._signatures["fun2"] == sig

    @jm.register("fun2b")
    def fun3(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    assert "fun3" not in jm._signatures
    assert jm._signatures["fun2b"] == sig


def test_use_register():
    jm = JitManager()

    sig = nt.float64(nt.int64, nt.float64)
    assert signature.is_numba_signature(sig)

    jm.register("simple", sig)

    @jm.njit_tmpl
    def simple(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    assert simple.nopython_signatures == [
        sig,
    ]

    @jm.njit_tmpl()
    def simple(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    assert simple.nopython_signatures == [
        sig,
    ]

    @jm.njit_tmpl("simple")
    def simpleb(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    assert simpleb.nopython_signatures == [
        sig,
    ]

    def simple(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    test = jm.njit_tmpl(simple)

    assert test.nopython_signatures == [
        sig,
    ]

    def test(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    test = jm.njit_tmpl("simple", test)

    assert test.nopython_signatures == [
        sig,
    ]
