"""
================================================
DFO Regressor for Subset Selection in a Pipeline
================================================

Including DFORegressor as part of a sklearn pipeline.

"""

import time

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

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
# Fit a RandomForestRegressor directly on the data
# ================================================
# We fit an out-of-the-box scikit-learn RF regressor
# on all of the original features of our training dataset
# Let's time how long this takes and the score (R² in this case)
# on our test dataset.

rf = RandomForestRegressor(500, random_state=42)

t0 = time.perf_counter()
rf.fit(X_train, y_train)
delta = time.perf_counter() - t0

rf_time = delta
rf_score = rf.score(X_test, y_test)

print(f"Random Forest fit time: {rf_time:.4f}")
print(f"Random Forest R² score on test set: {rf_score:.4f}")

# %%
# Use the discrete first-order method for subset selection as part of a Pipeline
# ==============================================================================
# Now we construct a scikit-learn pipeline that includes
# that includes a feature selection step based on the DFORegressor.

# using a very small selection threshold to select all features with non-zero coefs
pipeline = Pipeline(
    [
        (
            "selector",
            SelectFromModel(estimator=DFORegressor(k=10), threshold=1e-5),
        ),
        ("rf", RandomForestRegressor(500, random_state=42)),
    ]
)
t0 = time.perf_counter()
pipeline.fit(X_train, y_train)
delta = time.perf_counter() - t0

pipeline_time = delta
pipeline_score = pipeline.score(X_test, y_test)

print(f"DFO + RF Pipeline fit time: {pipeline_time:.4f}")
print(f"DFO + RF Pipeline R² score on test set: {pipeline_score:.4f}")
