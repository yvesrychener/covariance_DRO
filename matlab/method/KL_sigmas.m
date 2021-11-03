function [sigma_stars] = KL_sigmas(sigmas,dualvar)
%KL_SIGMAS Calculates the covariance eigenvalues for KL
%   sigmas: eigenvalues of sample covariance matrix
%   dualvr: optimal gamma
    sigma_stars = (sqrt(dualvar^2+16*dualvar.*sigmas.^2)-dualvar)./(8*sigmas);
end

