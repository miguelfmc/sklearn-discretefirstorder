"""
======================================
Discrete first-order method vs. Lasso
======================================

Comparison of the support recovery of discrete first-order method
and Lasso regression on a synthetic dataset.

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from discretefirstorder import DFORegressor

# %%
# Creating a synthetic dataset with some uninformative features
# ===============================================================
# We create a synthetic dataset with only 10 informative features.
# We keep the true coefficients for comparison with the estimates.

X, y, coef = make_regression(
    n_samples=10000,
    n_features=30,
    n_informative=10,
    coef=True,
    bias=5,
    noise=5,
    random_state=11,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# %%
# Lasso
# =====

lasso = Lasso()
lasso.fit(X_train, y_train)
lasso_score = lasso.score(X_test, y_test)
lasso_coef_error = np.sqrt(np.sum((lasso.coef_ - coef) ** 2))

# %%
# DFO
# ====

dfo = DFORegressor(k=10, fit_intercept=True, normalize=True)
dfo.fit(X_train, y_train)
dfo_score = dfo.score(X_test, y_test)
dfo_coef_error = np.sqrt(np.sum((dfo.coef_ - coef) ** 2))

# %%
# Comparison of the estimated coefficients
# ========================================
# We can see that the discrete first-order method, in this case,
# seems to recover the coefficients more faithfully to the ground truth

fig, ax = plt.subplots(1, 1, figsize=(9, 12))

index = np.arange(X.shape[1])
bar_width = 0.25

ax.barh(index, coef, bar_width, label="Ground Truth")
ax.barh(
    index + bar_width,
    dfo.coef_,
    bar_width,
    label=f"DFO - R2={dfo_score:.4f}; coef. error={dfo_coef_error:.2f}",
)
ax.barh(
    index + 2 * bar_width,
    lasso.coef_,
    bar_width,
    label=f"Lasso - R2={lasso_score:.4f}; coef. error={lasso_coef_error:.2f}",
)

ax.set_xlabel("Coefficient value")
ax.set_ylabel("Feature")
ax.set_title(
    "Lasso and DFO estimated regression coefficients compared to ground truth"
)
ax.set_yticks(index + bar_width)
ax.set_yticklabels(["feat_" + str(i) for i in range(X.shape[1])])
ax.grid(True)
ax.legend()

plt.show()
