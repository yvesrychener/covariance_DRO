function [sigma_stars] = WS_sigmas(sigmas,dualvar)
%WS_sigmas Calculates the covariance eigenvalues for wasserstein
%   sigmas: eigenvalues of sample covariance matrix
%   dualvr: optimal gamma
omegas = WS_omega(sigmas,dualvar);
sigma_stars = (omegas - dualvar./(6*omegas)).^2;
end

