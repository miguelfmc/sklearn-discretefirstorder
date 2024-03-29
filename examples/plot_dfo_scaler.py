"""
========================================
Scaling input data for the DFO Regressor
========================================

DFORegressor() can normalize the data internally or it can be used
in conjunction with a sklearn scaler e.g. StandardScaler.

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from discretefirstorder import DFORegressor

# %%
# Creating a synthetic dataset with some uninformative features
# ===============================================================
# We create a synthetic dataset with only 10 informative features.

means = np.random.randint(-10, 10, size=30)
stds = np.random.randint(1, 5, size=30)
cov = np.diag(stds**2)
epsilon = 5

np.random.seed(42)

X = np.random.multivariate_normal(means, cov, size=10000)
coef = np.concatenate(
    (np.array([10, -9, 8, -7, 6, -5, 4, -3, 2, -1]), np.zeros(20))
)
y = X @ coef + 5 * np.random.normal(size=10000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# %%
# Fit a DFORegressor directly on the data
# ========================================
# We fit the DFORegressor with fit_intercept=True and normalize=True
# to take care of data scaling internally.

dfo = DFORegressor(k=10, fit_intercept=True, normalize=True)
dfo.fit(X_train, y_train)

dfo_score = dfo.score(X_test, y_test)
dfo_coef_error = np.sqrt(np.sum((dfo.coef_ - coef) ** 2))

print(f"DFO R² score on test set: {dfo_score:.4f}")
print(f"DFO coef error: {dfo_coef_error:.4f}")

# %%
# Fit a DFORegressor in a Pipeline with StandardScaler
# ====================================================
# Now we first scale the data with StandardScaler and then fit the model
# with normalize=False.

pipeline_intercept = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("dfo", DFORegressor(k=10, fit_intercept=True, normalize=False)),
    ]
)
pipeline_intercept.fit(X_train, y_train)

# to compare with original coefficients we need to rescale the coefficients from the
# model
rescaled_coef = pipeline_intercept["dfo"].coef_ / X_train.std(axis=0)

pipeline_intercept_score = pipeline_intercept.score(X_test, y_test)
pipeline_intercept_coef_error = np.sqrt(np.sum((rescaled_coef - coef) ** 2))

print(
    f"DFO with external X scaling R² score on test set: {pipeline_intercept_score:.4f}"
)
print(
    f"DFO with external X scaling coef error: {pipeline_intercept_coef_error:.4f}"
)

# %%
# Fit a DFORegressor in a Pipeline with StandardScaler and no intercept
# =====================================================================
# If we don't want to fit an intercept term, we can use the same pipeline,
# setting fit_intercept=False and fitting the model on centered target data.

pipeline_no_intercept = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("dfo", DFORegressor(k=10, fit_intercept=False, normalize=False)),
    ]
)
pipeline_no_intercept.fit(X_train, (y_train - y_train.mean()))

# to compare with original coefficients we need to rescale the coefficients from the
# model
rescaled_coef = pipeline_no_intercept["dfo"].coef_ / X_train.std(axis=0)

pipeline_no_intercept_score = pipeline_no_intercept.score(X_test, y_test)
pipeline_no_intercept_coef_error = np.sqrt(np.sum((rescaled_coef - coef) ** 2))

print(
    f"DFO with external X scaling and no intercept R² score on test set: {pipeline_no_intercept_score:.4f}"
)
print(
    f"DFO with external X scaling and no intercept coef error: {pipeline_no_intercept_coef_error:.4f}"
)
