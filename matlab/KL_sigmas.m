function [sigma_stars] = KL_sigmas(sigmas,dualvar)
%KL_SIGMAS Summary of this function goes here
%   Detailed explanation goes here
    sigma_stars = (sqrt((dualvar./sigmas).^2+8*dualvar)-dualvar./sigmas)./4;
end

