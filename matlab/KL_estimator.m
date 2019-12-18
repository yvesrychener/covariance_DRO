function [sigma_star] = KL_estimator(sigma_hat,epsilon)
%KL_ESTIMATOR Summary of this function goes here
%   Detailed explanation goes here
[vecs, vals] = eig(sigma_hat);
vals = diag(vals);
p = length(vals);
sigma_max = max(vals);
interval = [0, 2*sigma_max^2*exp(-4*epsilon/p)/(1-exp(-2*epsilon/p))];

obj = @(dualvar) 2*epsilon + p + sum(log(KL_sigmas(vals, dualvar)./vals)-KL_sigmas(vals, dualvar)./vals);

dualvar_opt = bissection(obj, interval, 1e-10, 1e5);
vals_opt = KL_sigmas(vals, dualvar_opt);
sigma_star = vecs*diag(vals_opt)*transpose(vecs);
end

