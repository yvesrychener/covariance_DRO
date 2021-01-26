import numpy as np
from sklearn.covariance import EmpiricalCovariance

import covariance_DRO


def wasserstein(X, epsilon):
    return our_methods(X, epsilon, "Wasserstein")


def KL(X, epsilon):
    return our_methods(X, epsilon, "KLdirect")


def fisher_rao(X, epsilon):
    return our_methods(X, epsilon, "Fisher-Rao")


def our_methods(X, epsilon, method):
    c = EmpiricalCovariance(assume_centered=False).fit(X)
    cov = covariance_DRO.estimate_cov(c.covariance_, epsilon, method)
    return cov, c.location_


def linear_shrinkage(X, rho):
    c = EmpiricalCovariance(assume_centered=False).fit(X)
    emp_cov = c.covariance_
    mu = np.trace(emp_cov) / emp_cov.shape[0]
    cov = (1.0 - rho) * emp_cov
    cov.flat[:: emp_cov.shape[0] + 1] += rho * mu
    return cov, c.location_


def linear_shrinkage2(X, rho):
    c = EmpiricalCovariance(assume_centered=False).fit(X)
    emp_cov = c.covariance_
    mu = np.diag(np.diag(emp_cov))
    cov = (1.0 - rho) * emp_cov + mu * rho
    return cov, c.location_
