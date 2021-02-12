function [sigma_stars] = KL_sigmas(sigmas,dualvar)
%KL_SIGMAS Calculates the covariance eigenvalues for KL
%   sigmas: eigenvalues of sample covariance matrix
%   dualvr: optimal gamma
    sigma_stars = (sqrt((dualvar./sigmas).^2+8*dualvar)-dualvar./sigmas)./4;
end

