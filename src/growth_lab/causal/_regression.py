"""Minimal regression primitives shared by the causal estimators.

Implemented directly (no statsmodels/sklearn) so every number this package
reports has a visible derivation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class OlsFit:
    """OLS coefficients with homoskedastic standard errors."""

    beta: FloatArray
    se: FloatArray
    residuals: FloatArray


def ols(design: FloatArray, y: FloatArray) -> OlsFit:
    """Ordinary least squares; `design` must already include any intercept."""
    n, k = design.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than parameters ({k})")
    gram = design.T @ design
    beta: FloatArray = np.linalg.solve(gram, design.T @ y).astype(np.float64)
    residuals = y - design @ beta
    sigma2 = float(residuals @ residuals) / (n - k)
    cov = sigma2 * np.linalg.inv(gram)
    se = np.sqrt(np.diag(cov))
    return OlsFit(beta=beta, se=se, residuals=residuals)


def fit_logistic(
    design: FloatArray,
    y: FloatArray,
    ridge: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> FloatArray:
    """Logistic regression by Newton-Raphson (IRLS). Raises if not converged."""
    _, k = design.shape
    if set(np.unique(y)) - {0.0, 1.0}:
        raise ValueError("outcome must be binary 0/1")
    beta: FloatArray = np.zeros(k, dtype=np.float64)
    for _ in range(max_iter):
        p = sigmoid(design @ beta)
        weights = p * (1.0 - p) + 1e-12
        gradient = design.T @ (y - p) - ridge * beta
        hessian = (design * weights[:, None]).T @ design + ridge * np.eye(k)
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if float(np.max(np.abs(step))) < tol:
            return beta
    raise RuntimeError(f"logistic regression did not converge in {max_iter} iterations")


def sigmoid(eta: FloatArray) -> FloatArray:
    result: FloatArray = 1.0 / (1.0 + np.exp(-eta))
    return result


def add_intercept(x: FloatArray) -> FloatArray:
    if x.ndim == 1:
        x = x[:, None]
    result: FloatArray = np.column_stack([np.ones(len(x)), x])
    return result


def normal_two_sided_p(z: float) -> float:
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
