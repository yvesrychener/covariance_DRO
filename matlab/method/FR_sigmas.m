function [sigma_stars] = FR_sigmas(sigmas,dualvar)
%FR_SIGMAS Calculates the covariance eigenvalues for Fisher-Rao
%   sigmas: eigenvalues of sample covariance matrix
%   dualvr: optimal gamma
    sigma_stars = sigmas.*exp(-lambertw(2*sigmas.^2/dualvar)/2);
end

