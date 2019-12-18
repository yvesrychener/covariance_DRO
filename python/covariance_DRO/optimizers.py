# +-----------------------------------------------------------------------+
# | optimizers.py                                                         |
# | This module contains optimizers used by the estimators                |
# +-----------------------------------------------------------------------+


def bisection(f, interval, tol=1e-5, maxit=1e5):
    '''
    Perform the bisection algorithm for finding zero of a function
    
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
    
    '''
    # sanity check of inputs
    if not tol>0:
        raise ValueError('Tolerance does not follow necessary condition tol>0')
    if not maxit>0: 
        raise ValueError('maxit does not follow necessary condition maxit>0')
    # the bissection algorithm
    for i in range(int(maxit)):
        #stopping condition
        if(interval[1]-interval[0]<=2*tol):
            break
            
        # perform bissection
        middle = (interval[1]+interval[0])/2
        if f(middle)<0:
            #0 crossing occurs to the right of middle
            interval[0] = middle
        else:
            #0 crossing occurs to the left of middle
            interval[1]=middle        
    return (interval[1]+interval[0])/2