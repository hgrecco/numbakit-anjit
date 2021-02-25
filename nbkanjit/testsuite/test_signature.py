import inspect

import pytest
from numba import njit
from numba import types as nt

from nbkanjit import exceptions, signature


def test_is_numba_type():
    assert signature.is_numba_type(nt.void)
    assert signature.is_numba_type(nt.float64)
    assert signature.is_numba_type(nt.float64[:])
    assert signature.is_numba_type(nt.float64[::])
    assert signature.is_numba_type(nt.float64[:, :])
    assert signature.is_numba_type(nt.List(int))
    assert signature.is_numba_type(nt.UniTuple(int, 3))
    assert not signature.is_numba_type(None)
    assert not signature.is_numba_type(int)
    assert not signature.is_numba_type(float)


def test_verify_mapping():

    assert signature.verify_mapping(dict(a=nt.float64)) == tuple()

    assert signature.verify_mapping(dict(a=888), raise_on_err=False) == (("a", 888),)

    with pytest.raises(exceptions.NotNumbaSignature, match=r".*888.*"):
        signature.verify_mapping(dict(a=888))


def test_build_signature():
    def fun(x: int, y: float) -> float:
        pass

    assert signature.build_signature(fun, signature.DEFAULT_TYPE_MAPPING) != nt.float64(
        nt.int64, nt.int64
    )
    assert signature.build_signature(fun, signature.DEFAULT_TYPE_MAPPING) == nt.float64(
        nt.int64, nt.float64
    )

    def fun(x: int, y: float) -> None:
        pass

    assert signature.build_signature(fun, signature.DEFAULT_TYPE_MAPPING) == nt.void(
        nt.int64, nt.float64
    )

    def fun(x: int, y: nt.float64) -> None:
        pass

    assert signature.build_signature(fun, signature.DEFAULT_TYPE_MAPPING) == nt.void(
        nt.int64, nt.float64
    )

    def fun(x: int, y: nt.float64[:]) -> None:
        pass

    assert signature.build_signature(fun, signature.DEFAULT_TYPE_MAPPING) == nt.void(
        nt.int64, nt.float64[:]
    )

    def fun(x: int, y: 888) -> None:
        pass

    with pytest.raises(exceptions.UnknownAnnotation, match=r".*888.*"):
        signature.build_signature(fun, signature.DEFAULT_TYPE_MAPPING)


def test_missing_args():
    def fun(x: int, y) -> float:
        pass

    assert signature.build_signature(
        fun, signature.DEFAULT_TYPE_MAPPING, on_missing_arg=int
    ) == nt.float64(nt.int64, nt.int64)

    with pytest.raises(exceptions.MissingAnnotation, match=r".*y.*"):
        signature.build_signature(fun, signature.DEFAULT_TYPE_MAPPING)


def test_missing_rettype():
    def fun(x: int, y: int):
        pass

    assert signature.build_signature(
        fun, signature.DEFAULT_TYPE_MAPPING, on_missing_ret=float
    ) == nt.float64(nt.int64, nt.int64)

    with pytest.raises(exceptions.MissingAnnotation, match=r".*return.*"):
        signature.build_signature(fun, signature.DEFAULT_TYPE_MAPPING)


def test_numba_types():
    def fun(x: nt.float64, y: nt.float64) -> nt.float64:
        pass

    assert signature.build_signature(fun, {}) == nt.float64(nt.float64, nt.float64)


def test_custom_mapping():
    d = dict(a=nt.float64)

    def fun(x: "a", y: "a") -> "a":  # noqa: F821
        pass

    assert signature.build_signature(fun, d) == nt.float64(nt.float64, nt.float64)

    def fun(x: int):
        pass

    with pytest.raises(exceptions.UnknownAnnotation):
        signature.build_signature(fun, d)


def test_anjit():
    def _fun1(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    @signature.anjit
    def fun1a(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    @signature.anjit
    def fun1b(x: int, y: float) -> float:
        return x + y

    @njit(nt.float64(nt.int64, nt.float64))
    def fun2(x, y):
        return x + y

    assert fun1a.__name__ == "fun1a"
    assert fun1a.__annotations__ == _fun1.__annotations__
    assert fun1a.__doc__ == _fun1.__doc__
    assert inspect.signature(fun1a) == inspect.signature(_fun1)
    assert fun1a.nopython_signatures
    assert fun1a.nopython_signatures == fun1b.nopython_signatures
    assert fun1a.nopython_signatures == fun2.nopython_signatures


def test_anjit_custom_mapping():

    d = {**signature.DEFAULT_TYPE_MAPPING, "a": nt.float64}

    @signature.anjit(mapping=d)
    def fun1(x: int, y: "a") -> float:  # noqa: F821
        """Simple doc"""
        return x + y

    @njit(nt.float64(nt.int64, nt.float64))
    def fun2(x, y):
        return x + y

    assert fun1.nopython_signatures
    assert fun1.nopython_signatures == fun2.nopython_signatures


def test_function_wrapper():
    @signature.anjit
    def fun1a(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    @signature.anjit
    def fun1b(x: int, f: signature.Function(fun1a)) -> float:
        return f(x, 1.0 * x)

    ft = nt.float64(nt.int64, nt.FunctionType(nt.float64(nt.int64, nt.float64)))
    assert fun1b.nopython_signatures == [ft]


def test_function_argument_wrapper():
    @signature.anjit
    def fun1a(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    @signature.anjit
    def fun1b(x: int, y: signature.Function(fun1a).y) -> float:
        return x + y

    @njit(nt.float64(nt.int64, nt.float64))
    def fun2(x, y):
        return x + y

    assert fun1a.nopython_signatures == fun1b.nopython_signatures
    assert fun1a.nopython_signatures == fun2.nopython_signatures


def test_function_return_wrapper():
    @signature.anjit
    def fun1a(x: int, y: float) -> float:
        """Simple doc"""
        return x + y

    @signature.anjit
    def fun1b(x: int, y: signature.Function(fun1a)._return) -> float:
        return x + y

    @njit(nt.float64(nt.int64, nt.float64))
    def fun2(x, y):
        return x + y

    assert fun1a.nopython_signatures == fun1b.nopython_signatures
    assert fun1a.nopython_signatures == fun2.nopython_signatures
