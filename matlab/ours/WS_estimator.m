function [sigma_star] = WS_estimator(sigma_hat,epsilon)
%WS_estimator Calculate shrinkage covariance matrix using Wasserstein method
[vecs, vals] = eig(sigma_hat);
vals = diag(vals);

% objective function
obj = @(dualvar) epsilon^2 - sum((sqrt(vals)-sqrt(WS_sigmas(vals,dualvar))).^2);
% find bisection interval
left = 0;
right = 1;
for i=1:1e5
    if obj(right)>0
        break;
    end
    right = right*2;
end
interval = [left, right];
dualvar_opt = bissection(obj, interval, 1e-5, 1e5);
vals_opt = WS_sigmas(vals, dualvar_opt);
sigma_star = vecs*diag(vals_opt)*transpose(vecs);
end

