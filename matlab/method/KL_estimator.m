function [sigma_star] = KL_estimator(sigma_hat,epsilon)
%KL_ESTIMATOR Calculate shrinkage covariance matrix using KL method
[vecs, vals] = eig(sigma_hat);
vals = diag(vals);
p = length(vals);
sigma_max = max(vals);
interval = [0, 4*sigma_max^2*exp(-4*epsilon/p)/(1-exp(-2*epsilon/p))];
X = @(dualvar) 2./(1+sqrt(1+16.*vals.^2./dualvar));

obj = @(dualvar) 2*epsilon + p + sum(log(X(dualvar))-X(dualvar));

dualvar_opt = bissection(obj, interval, 1e-10, 1e5);
vals_opt = KL_sigmas(vals, dualvar_opt);
sigma_star = vecs*diag(vals_opt)*transpose(vecs);
end

