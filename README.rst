.. -*- mode: rst -*-

|ReadTheDocs|_ |Maintenance yes|

.. |ReadTheDocs| image:: https://readthedocs.org/projects/sklearn-firstordersubset/badge/?version=latest
.. _ReadTheDocs: https://sklearn-firstordersubset.readthedocs.io/en/latest/?badge=latest

.. |Maintenance yes| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://github.com/miguelfmc/sklearn-discretefirstorder/commit-activity

A Discrete First Order Method for Subset Selection
==================================================

.. _scikit-learn: https://scikit-learn.org
.. _documentation: https://sklearn-discretefirstorder.readthedocs.io/en/latest/quick_start.html

**sklearn-discretefirstorder** is a light-weight package that implements a simple
discrete first-order method for best feature subset selection in linear regression.

The discrete first-order method is based on the technique described by Berstimas et al. [1]_

The package is built on top of the scikit-learn_ framework and is compatible with scikit-learn methods
such as cross-validation and pipelines.
I followed the guidelines for developing scikit-learn estimators
as outlined in the `scikit-learn documentation <https://scikit-learn.org/stable/developers/develop.html>`_.

Installation
------------

To install the package, clone this repo and run `pip install`.
   
      $ git clone https://github.com/miguelfmc/sklearn-discretefirstorder
      $ cd sklearn-discretefirstorder
      $ pip install .

from the root directory.

Quick Start
-----------

Once you have installed the package you can start using it as follows.

The key estimator in this packages is the
:class:`discretefirstorder.DFORegressor`.
You can import it as::

    >>> from discretefirstorder import DFORegressor

Easily fit the estimator as follows::

    >>> import numpy as np
    >>> from discretefirstorder import DFORegressor
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.random.normal(size=(100, ))
    >>> estimator = DFORegressor(k=5)
    >>> estimator.fit(X, y)

For more examples, see the documentation_.

Known Issues
------------
This package is still at a very early stage of development. The following issues are known:
- Optimization routines are implemented in Python, which makes them slow.
- At the moment, the package only supports squared error loss minimization but there are plans to include support for absolute error loss minimization.
- At the moment, there is no support for classification problems i.e. logistic regression.


.. [1] Dimitris Bertsimas. Angela King. Rahul Mazumder. "Best subset selection via a modern optimization lens." Ann. Statist. 44 (2) 813 - 852, April 2016. https://doi.org/10.1214/15-AOS1388 
