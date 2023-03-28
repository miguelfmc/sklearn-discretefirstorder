.. title:: User guide : contents

.. _user_guide:

============================================================
User guide: discrete first-order method for subset selection
============================================================

The DFO Regressor
-----------------

The key estimator in this packages is the
:class:`discretefirstorder.DFORegressor`.
You can import it as::

    >>> from discretefirstorder import DFORegressor

Fitting the estimator
---------------------

Easily fit the estimator as follows::

    >>> import numpy as np
    >>> from discretefirstorder import DFORegressor
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.random.normal(size=(100, ))
    >>> estimator = DFORegressor()
    >>> estimator.fit(X, y)
