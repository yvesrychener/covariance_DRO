# +-----------------------------------------------------------------------+
# | estimators.py                                                         |
# | This module contains estimators                                       |
# +-----------------------------------------------------------------------+
import numpy as np
from scipy.special import lambertw
from warnings import warn
from . import optimizers


def kl_direct(sigma_hat, epsilon, tol=1e-5, maxit=1e5):
    """
    Calculate Sigma_star using Kullback Leibler direct method
    
    Parameters
    ----------
    sigma_hat : numpy.ndarray
        the sample covariance matrix
    epsilon : float
        the radius of the uncertainty ball 
        'KLdirect', ...
    tol : float, optional
        stopping criterion for optimization
    maxit : float or int, optional
        maximum number of iterations for optimization
        
    Returns
    -------
    sigma_star : numpy ndarray
        the estimated covariance matrix
    """
    # eigenvalue decomposition
    w, v = np.linalg.eig(sigma_hat)

    p = w.size
    # definition of functions and bissection
    # definition of sigma_star(gamma)
    sigma = lambda gamma: (np.sqrt((gamma ** 2) / (w ** 2) + 8 * gamma) - gamma / w) / 4
    # definition of sigma_star_prime(gamma)
    sigma_p = lambda gamma: (
        gamma * (w ** (-2)) + 4 - np.sqrt((gamma / w) ** 2 + 8 * gamma) / w
    ) / (4 * np.sqrt((gamma / w) ** 2 + 8 * gamma))
    # definition of f_prime(gamma)
    f_prime = (
        lambda gamma: (2 * epsilon + p)
        + (np.log(sigma(gamma) / w) - sigma(gamma) / w).sum()
    )
    # (2*epsilon-np.log(w).sum())+\
    #               (2*sigma(gamma)*sigma_p(gamma)+np.log(sigma(gamma))+gamma*(sigma_p(gamma)/sigma(gamma))).sum()
    # bisection interval
    p = sigma_hat.shape[0]
    interval = [
        0,
        (2 * w.max() ** 2 * np.exp(-4 * epsilon / p)) / (1 - np.exp(-2 * epsilon / p)),
    ]

    # find the optimal gamma
    gamma_star = optimizers.bisection(f_prime, interval, tol=tol, maxit=maxit)

    # calculate the estimated covariance matrix
    w_new = sigma(gamma_star)
    Sigma_star = v @ np.diag(w_new) @ v.transpose()
    return Sigma_star


def wasserstein(sigma_hat, epsilon, tol=1e-5, maxit=1e5):
    """
    Calculate Sigma_star using Wasserstein method
    
    Parameters
    ----------
    sigma_hat : numpy.ndarray
        the sample covariance matrix
    epsilon : float
        the radius of the uncertainty ball 
        'KLdirect', ...
    tol : float, optional
        stopping criterion for optimization
    maxit : float or int, optional
        maximum number of iterations for optimization
        
    Returns
    -------
    sigma_star : numpy ndarray
        the estimated covariance matrix
    """
    if epsilon>np.trace(sigma_hat):
        warn('Epsilon bigger than max value')
        epsilon = np.trace(sigma_hat)
    # eigenvalue decomposition
    w, v = np.linalg.eig(sigma_hat)
    # definition of functions and bissection
    # definition of sigma
    omega = lambda gamma: np.cbrt(
        (gamma / 4) * (np.sqrt(w) + np.sqrt(w + 2 * gamma / 27))
    )
    sigma = lambda gamma: (omega(gamma) - gamma / (6 * omega(gamma))) ** 2

    # definition of f_prime
    f_prime = (
        lambda gamma: epsilon ** 2 - ((np.sqrt(w) - np.sqrt(sigma(gamma))) ** 2).sum()
    )

    # find the bisection interval
    left = 0
    # right = 1
    # for i in range(int(maxit)):
    #    if f_prime(right)>0:
    #        break
    #    right = right*2
    p = len(w)
    sigma_p = max(w)
    right = 2 * (np.sqrt(p * sigma_p) - epsilon) ** 3 / (p * epsilon)
    interval = [left, right]

    # find optima gamma
    gamma_star = optimizers.bisection(f_prime, interval, tol=tol, maxit=maxit)
    # calculate the estimated covariance matrix
    w_new = sigma(gamma_star)
    Sigma_star = v @ np.diag(w_new) @ v.transpose()
    return Sigma_star


def fisher_rao(sigma_hat, epsilon, tol=1e-5, maxit=1e5):
    """
    Calculate Sigma_star using Fisher-Rao method
    
    Parameters
    ----------
    sigma_hat : numpy.ndarray
        the sample covariance matrix
    epsilon : float
        the radius of the uncertainty ball 
        'KLdirect', ...
    tol : float, optional
        stopping criterion for optimization
    maxit : float or int, optional
        maximum number of iterations for optimization
        
    Returns
    -------
    sigma_star : numpy ndarray
        the estimated covariance matrix
    """
    # eigenvalue decomposition
    w, v = np.linalg.eig(sigma_hat)
    # definition of functions and bissection
    # definition of sigma
    sigma = lambda gamma: w * np.exp(-1 / 2 * lambertw((2 * w ** 2) / (gamma)) ** 2)

    # definition of f_prime
    f_prime = (
        lambda gamma: 4 * epsilon ** 2 - (lambertw((2 * w ** 2) / (gamma)) ** 2).sum()
    )

    # find the bisection interval
    left = 0
    right = np.linalg.norm(sigma_hat, ord="fro") ** 2 / epsilon
    interval = [left, right]

    # find optima gamma
    gamma_star = optimizers.bisection(f_prime, interval, tol=tol, maxit=maxit)
    # calculate the estimated covariance matrix
    w_new = sigma(gamma_star)
    Sigma_star = v @ np.diag(w_new) @ v.transpose()
    if not np.all(np.isreal(Sigma_star)):
        warn("Calculated Covariance matrix has complex terms")
        return Sigma_star
    else:
        return np.real(Sigma_star)


def estimate_cov(sigma_hat, epsilon, method, tol=1e-5, maxit=1e5):
    """
    Estimate Covariance matrix using distributionally robust optimization
    
    Parameters
    ----------
    sigma_hat : numpy.ndarray
        the sample covariance matrix
    epsilon : float
        the radius of the uncertainty ball 
    method : string
        the method to use. must be one of the following:
        'KLdirect', ...
    tol : float, optional
        stopping criterion for optimization
    maxit : float or int, optional
        maximum number of iterations for optimization
        
    Raises
    ------
    ValueError
        negative ball radius
    ValueError
        invalid method
    ValueError
        non-positive-semidefinite covariance matrix
        
    Returns
    -------
    sigma_star : numpy ndarray
        the estimated covariance matrix
    """

    # lookup table for functions
    function_dict = {
        "KLdirect": kl_direct,
        "Wasserstein": wasserstein,
        "Fisher-Rao": fisher_rao,
    }

    # sanity check for inputs
    if not epsilon > 0:
        raise ValueError("ball radius must be >0")
    if not np.all(np.linalg.eigvals(sigma_hat) >= 0):
        raise ValueError("non-positive-semidefinite covariance matrix")
    if not method in list(function_dict.keys()):
        raise ValueError("invalid method")

    # perform the estimation and return the result
    return function_dict[method](sigma_hat, epsilon, tol, maxit)
