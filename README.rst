.. image:: https://img.shields.io/pypi/v/numbakit-anjit.svg
    :target: https://pypi.python.org/pypi/numbakit-anjit
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/l/numbakit-anjit.svg
    :target: https://pypi.python.org/pypi/numbakit-anjit
    :alt: License

.. image:: https://img.shields.io/pypi/pyversions/numbakit-anjit.svg
    :target: https://pypi.python.org/pypi/numbakit-anjit
    :alt: Python Versions

.. image:: https://github.com/hgrecco/numbakit-anjit/workflows/CI/badge.svg?branch=main
    :target: https://github.com/hgrecco/numbakit-anjit/actions?query=workflow%3ACI

.. image:: https://github.com/hgrecco/numbakit-anjit/workflows/Lint/badge.svg?branch=main
    :target: https://github.com/hgrecco/numbakit-anjit/actions?query=workflow%3ALint

.. image:: https://coveralls.io/repos/github/hgrecco/numbakit-anjit/badge.svg?branch=main
    :target: https://coveralls.io/github/hgrecco/numbakit-anjit?branch=main

.. image:: https://readthedocs.org/projects/numbakit-anjit/badge/
    :target: http://numbakit-anjit.readthedocs.org/
    :alt: Docs


numbakit-anjit: leveraging numba to speed up ODE integration
============================================================

numbakit-anjit (nbkanjit) to assist Numba_ intensive project
by providing `anjit`, an annotation aware numba jit decorator.

It runs in Python 3.7+ depending on Numba_. It is licensed under
BSD.

It is extremely easy and natural to use:

.. code-block:: python

    >>> from numba import types as nt
    >>> import nbkanjit
    >>> @nbkanjit.anjit
    ... def func(x: nt.float64, y: nt.float64) -> nt.float64:
    ...     return x + y

You can also use Python types:

.. code-block:: python

    >>> @nbkanjit.anjit
    ... def func(x: float, y: float) -> float:
    ...     return x + y

which are mapped to numba types.

You can use

.. code-block:: python

    >>> from nbkanjit import Function as F_
    >>> @nbkanjit.anjit
    ... def func1(x: int, y: float) -> float:
    ...     return x + y
    >>> def func2(x: int, y: F_(funct1)._return) -> float:
    ...     return x + y

You can also use the annotation of any argument. For example,
`F_(func).x` in this case is equivalent to `int`. Or even the
full function `F_(func)` that will return
`FunctionType(float64(int, float64))`

It also provides a manager to encapsulate (and reuse different parameters)

.. code-block:: python

    >>> import nbkanjit
    >>> jm = nbkanjit.JitManager(cache=True)
    >>> @jm.anjit
    ... def func(x: float, y:float) -> nt.float64:
    ...     return x + y

even to be applied in to the standard numba njit.

    >>> jm = nbkanjit.JitManager(cache=True)
    >>> @jm.njit
    ... def func(x, y):
    ...     return x + y

And you can teach the manager new tricks:

    >>> jm.mapping["array1d"] = nt.float64[:]

by mapping any python object into a numba type.


Quick Installation
------------------

To install numbakit-anjit, simply (*soon*):

.. code-block:: bash

    $ pip install numbakit-anjit

or utilizing conda, with the conda-forge channel (*soon*):

.. code-block:: bash

    $ conda install -c conda-forge numbakit-anjit

and then simply enjoy it!


Why
---

Numba `njit` is awesome. Simple to use, produces the appropriate machine code
once that the function is called. As the `Numba docs`_ says:

.. note::

   in [Lazy mode], compilation will be deferred until the first function
   execution. Numba will infer the argument types at call time, and
   generate optimized code based on this information. Numba will also
   be able to compile separate specializations depending on the input
   types.

But numba also has an **eager mode**:

.. note::

   In which you can also tell Numba the function signature you are expecting.
   [..] In this case, the corresponding specialization will be compiled by the
   decorator, and no other specialization will be allowed. This is useful
   if you want fine-grained control over types chosen by the compiler (for
   example, to use single-precision floats).

This can produce slightly faster code as the compiler does not need to infer
the types. It also provides type check at definition time ensuring correctness.
In numba intensive projects, this can be an useful trait. Finally, eager
compilation is currently required to have two functions with the same signature
to be arguments of a third one, without needing to recompile this last one in each
case.

While developing `numbakit-ode`_ I was missing two things:

1. That eager compilation make use of function annotations
2. A global manager object to manipulate in one place numba
   jit options.

So, `numbakit-anjit` was born.


----

numbakit-anjit is maintained by a community. See AUTHORS_ for a complete list.

To review an ordered list of notable changes for each version of a project,
see CHANGES_


.. _`Numba`: https://numba.pydata.org/
.. _`AUTHORS`: https://github.com/hgrecco/numbakit-anjit/blob/master/AUTHORS
.. _`CHANGES`: https://github.com/hgrecco/numbakit-anjit/blob/master/CHANGES
.. _`Numba docs`: https://numba.pydata.org/numba-doc/latest/user/jit.html#compiling-python-code-with-jit
.. _`numbakit-ode`: https://github.com/hgrecco/numbakit-ode