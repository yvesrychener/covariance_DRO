# +-----------------------------------------------------------------------+
# | optimizers.py                                                         |
# | This module contains optimizers used by the estimators                |
# +-----------------------------------------------------------------------+
from warnings import warn


def bisection(f, interval, tol=1e-5, maxit=1e5):
    """
    Perform the bisection algorithm for finding zero of an increasing function
    
    Parameters
    ----------
    f : function handle
        the function for which we want to find the zero
    interval : list with 2 float elements
        left and right boundary for starting the algorithm, must be of different sign
    tol : float, optional
        stopping criterion, error_in_x(solution)<=tol
    maxit : float or int, optional
        maximum number of iterations
        
    Raises
    ------
    ValueError
        negative tolerance
    ValueError
        non-positive maxit
        
    Returns
    -------
    zero : float
        the x position of the zero found
    
    """
    # sanity check of inputs
    left = interval[0]
    right = interval[1]
    if not tol > 0:
        raise ValueError("Tolerance does not follow necessary condition tol>0")
    if not maxit > 0:
        raise ValueError("maxit does not follow necessary condition maxit>0")
    # the bissection algorithm
    for i in range(int(maxit)):
        # stopping condition
        if right - left <= 2 * tol:
            break

        # perform bissection
        middle = (left + right) / 2
        if f(middle) < 0:
            # 0 crossing occurs to the right of middle
            left = middle
        else:
            # 0 crossing occurs to the left of middle
            right = middle
    res = (left + right) / 2
    if res - interval[0] < tol:
        warn(
            "Result is on the left side of interval, which may indicate an incorrect interval."
        )
    if interval[1] - res < tol:
        warn(
            "Result is on the right side of interval, which may indicate an incorrect interval."
        )
    return res

