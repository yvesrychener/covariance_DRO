function [COV, epsilon] = gaussian_likelyhood_CV(estim,X, epsilons)
%UNTITLED Find the best covariance matrix with gaussian likelyhood
%leave-1-out cross-validataion
log_likelyhoods = zeros(size(epsilons));
for i = 1:length(epsilons)
    for j = 1:size(X,1)
        % split into train and validation
        x_valid = X(j,:);
        X_train = X;
        X_train(j,:) = [];
        % find the covariance matrix 
        c = estim(cov(X_train), epsilons(i));
        % use the gaussian log-likelyhood as score
        log_likelyhoods(i) = log_likelyhoods(i) - det(c)/2 - x_valid*pinv(c)*transpose(x_valid)/2;
    end
end
[argvalue, argmax] = max(log_likelyhoods);
epsilon = epsilons(argmax);
COV = estim(cov(X), epsilon);
end

