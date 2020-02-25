function [res] = simulations(covariance, loss, n_tries, n_samples, eps_wass, eps_KL)
%simulations Runs simulations of the performance for a given true covariance matrix
% samples n_samples from a normal distribution with covariance matrix
% covaricance, computes the loss
% returns a matrix n_tries, n_estimator of losses
% currently, the order of the estimators is:
%   sample covariance
%   LW
%   RBLW
%   OAS
%   KL
%   Wasserstein
l = @(M) loss(M,covariance);
res = zeros(n_tries, 6);
for i=1:n_tries
    % calculate the covariance matrices
    X = mvnrnd(zeros(1, size(covariance,1)), covariance, n_samples);
    [LW, RBLW, OAS] = benchmark_matrices(X);
    COV = cov(X);
    WS = WS_estimator(COV, eps_wass);
    KL = KL_estimator(COV, eps_KL);
    % add the losses to the table
    res(1,:) = [l(COV), l(LW), l(RBLW), l(OAS), l(KL), l(WS)];
end
end

