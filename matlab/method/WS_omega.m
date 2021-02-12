function [omegas] = WS_omega(sigmas,dualvar)
%WS_omega Calculates omegas (helping variable) for wasserstein
%   sigmas: eigenvalues of sample covariance matrix
%   dualvr: optimal gamma
    omegas = ((dualvar/4)*(sqrt(sigmas)+sqrt(sigmas + 2*dualvar/27))).^(1/3);
end

