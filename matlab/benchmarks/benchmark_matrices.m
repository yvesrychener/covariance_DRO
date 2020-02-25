function [LW, RBLW, OAS] = benchmark_matrices(X)
%benchmark_matrices Returns the covariance estimates of the benchmarks
%   X : data NxP : N : samples P: features
%   Returns:
%   LW: Ledoit-Wolf Estimator (Ledoit-Wolf [2004], well-conditioned estimator for large-dimensional covariance matrices)
%   RBLW: Rao-Blackwell LW Estimator (Chen [2010], Shrinkage Algorithms for MMSE Covariance Estimation)
%   OAS: Oracle Approximation Shrinkage (Chen [2010], Shrinkage Algorithms for MMSE Covariance Estimation)

% path organization
path = fileparts(mfilename('fullpath'));
addpath(strcat(path, '\lw'));
addpath(strcat(path, '\oas_and_rblw'));
% calculate the matrices
LW = cov1para(X);
RBLW = shrinkage_cov(X,'rblw');
OAS = shrinkage_cov(X, 'oas');
end

