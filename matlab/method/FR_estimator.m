function [sigma_star] = FR_estimator(sigma_hat,epsilon)
%FR_ESTIMATOR Calculate shrinkage covariance matrix using Fisher-Rao method
[vecs, vals] = eig(sigma_hat);
vals = diag(vals);
interval = [0, norm(sigma_hat,'fro')^2/epsilon];

obj = @(dualvar) 4*epsilon^2-sum(lambertw(2*vals.^2/dualvar).^2);
dualvar_opt = bissection(obj, interval, 1e-10, 1e5);
vals_opt = FR_sigmas(vals, dualvar_opt);
sigma_star = vecs*diag(vals_opt)*transpose(vecs);
end

