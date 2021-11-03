import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from warnings import warn
import warnings

import covariance_DRO
import nl_shrinkage

warnings.filterwarnings("ignore")

import matlab.engine
eng = matlab.engine.start_matlab()
eng.addwise(nargout=0)
def numpy_to_matlab(array):
    return matlab.double(array.tolist())
nl_LW = nl_shrinkage.MatlabShrinkage()

# leave-1-out CV for selecting ball radius or other hyperparameters
def l1o_cv(X, method, epsilon_candidates):
    best_epsilon, best_std = None, float('inf')
    for e in epsilon_candidates:
        try:
            returns = []
            for i in range(X.shape[0]):
                # train-validation split
                val = np.expand_dims(X[i], 0)
                train = X[np.arange(0, X.shape[0], 1) != i]
                # find the weight
                cov_inv = np.linalg.inv(method(train, e))
                ones = np.ones((cov_inv.shape[0],1))
                w = (cov_inv@ones)/(ones.T@cov_inv@ones)
                # calculate portfolio returns and metrics
                returns.append(val@w)
            # keep track of best epsilon
            std = np.std(returns)
            mean = np.mean(returns)
            sharpe = mean/std
            if std < best_std:
                best_std = np.std(returns)
                best_epsilon = e
        except np.linalg.LinAlgError:
            print(f'LinalgError for Epsilon={e}')
            continue
    
    # warn if best epsilon is on border
    if best_epsilon==epsilon_candidates[0]:
        warn('Epsilon on left side of interval')
    if best_epsilon==epsilon_candidates[-1]:
        warn('Epsilon on right side of interval')
    # return estimated covariance matrix
    return method(X, best_epsilon)

# Covariance estimators:

# linear method
def linearMethod(X, epsilon):
    c = EmpiricalCovariance(assume_centered=False).fit(X)
    emp_cov = c.covariance_
    mu = np.trace(emp_cov) / emp_cov.shape[0]
    cov = (1.0 - epsilon) * emp_cov
    cov.flat[:: emp_cov.shape[0] + 1] += epsilon * mu
    return cov

# linear method with diagonal target
def linearMethodDiagonal(X, epsilon):
    c = EmpiricalCovariance(assume_centered=False).fit(X)
    emp_cov = c.covariance_
    mu = np.diag(np.diag(emp_cov))
    cov = (1.0 - epsilon) * emp_cov + mu * epsilon
    return cov

# our methods
def our_methods(X, epsilon, method):
    c = EmpiricalCovariance(assume_centered=False).fit(X)
    cov = covariance_DRO.estimate_cov(c.covariance_, epsilon, method)
    return cov

# Wasserstein inverse shrinkage
def WISE(X, epsilon):
    c = EmpiricalCovariance(assume_centered=False).fit(X)
    res = eng.wise(numpy_to_matlab(c.covariance_), float(epsilon))
    cov = np.linalg.inv(np.asarray(res['value']))
    return cov

# rolling portfolio tester: estimate covariance from last 50 months, compute minimum variance portfolio
# uese computed portfolio for the next "step" months
def portfolio_tester_rolling(universe, cov_estimator, verbose=False, step=12):
    returns = None
    for i in range(50,len(universe)-step+1, step):
        train = universe.loc[i-50:i-1].drop('Date', axis=1).to_numpy()
        test = universe.loc[i:i+step-1].drop('Date', axis=1).to_numpy()
        # estimate covariance matrix and asset allocation
        cov_inv = np.linalg.inv(cov_estimator(train))
        ones = np.ones((cov_inv.shape[0],1))
        w = (cov_inv@ones)/(ones.T@cov_inv@ones)
        # calculate portfolio returns and metrics
        portfolio_returns = test@w
        if returns is None:
            returns = portfolio_returns.flatten()
        else:
            returns = np.hstack([returns, portfolio_returns.flatten()])
        dates = universe['Date'][step:i+2*step]
    std = returns.std()
    mean = returns.mean()
    sharpe = mean/std
    if verbose:
        print('Mean: {}'.format(mean))
        print('Std: {}'.format(std))
        print('Sharpe: {}'.format(sharpe))
    return {
        'mean' : mean,
        'std' : std,
        'sharpe' : sharpe
    }

# run portfolio optimization
def run_po(stride):
    # load the investment universe, clean the data and exclude NaN values
    universe = pd.read_csv('data/48_Industry_Portfolios.CSV', header=6, nrows=1140)
    universe = universe.replace(-99.99, np.NaN)
    universe = universe.rename(columns = {'Unnamed: 0':'Date'})
    universe = universe[516:]

    # exclude first months such that investment starts in januray 1974
    universe = universe.reset_index(drop=True)
    universe = universe[4:]
    universe = universe.reset_index(drop=True)

    # compute the portfolio statistics using the different covariance estimation methods
    res_empirical = portfolio_tester_rolling(universe, 
                            lambda X: EmpiricalCovariance(assume_centered=False).fit(X).covariance_, step=stride)
    res_linear = portfolio_tester_rolling(universe, 
                            lambda X: l1o_cv(X, linearMethod, np.logspace(-5,0,50)), step=stride)
    res_lineardiag = portfolio_tester_rolling(universe, 
                            lambda X: l1o_cv(X, linearMethodDiagonal, np.logspace(-5,0,50)), step=stride)
    res_ws = portfolio_tester_rolling(universe, 
                            lambda X: l1o_cv(X, lambda X, epsilon: our_methods(X, epsilon, 'Wasserstein'), np.logspace(-5,4,50)), step=stride)
    res_kl = portfolio_tester_rolling(universe, 
                            lambda X: l1o_cv(X, lambda X, epsilon: our_methods(X, epsilon, 'KLdirect'), np.logspace(-5,2,50)), step=stride)
    res_fr = portfolio_tester_rolling(universe, 
                            lambda X: l1o_cv(X, lambda X, epsilon: our_methods(X, epsilon, 'Fisher-Rao'), np.logspace(-5,2,50)), step=stride)
    res_wise = portfolio_tester_rolling(universe, 
                            lambda X: l1o_cv(X, lambda X, epsilon: WISE(X, epsilon), np.logspace(-5,3,50)), step=stride)
    res_nllw = portfolio_tester_rolling(universe, 
                            lambda X: nl_LW.nl_shrinkage(X), step=stride)
    
    # build the results and return them
    for res, name in zip([res_empirical, res_linear, res_lineardiag, res_ws, res_kl, res_fr, res_wise, res_nllw],
                         ['Empirical', 'Linear', 'Linear Diagonal', 'Wasserstein', 'KL', 'Fisher-Rao', 'WISE', 'NLLW']):
        res['stride'] = stride
        res['method'] = name
    return [res_empirical, res_linear, res_lineardiag, res_ws, res_kl, res_fr, res_wise, res_nllw]

if __name__=='__main__':
    results = []
    for stride in [1, 3, 6, 12, 24, 36, 48, 50, 60]:
        print(f'Running for Stride {stride}')
        results = results + run_po(stride)
        # break in testing mode
    df = pd.DataFrame(results)
    df.to_csv('results_fama48_stride.csv')