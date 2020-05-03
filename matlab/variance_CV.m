function [COV, epsilon] = variance_CV(estim,X, epsilons, k)
%UNTITLED Find the best covariance matrix for minimum variance using
%k-fold-out cross-validataion
means = zeros(size(epsilons));
l = ceil(size(X,1)/k);
for i = 1:length(epsilons)
    for j = 0:(k-1)
        % split into train and validation
        x_valid = X((1+j*l):min(l*(j+1), size(X,1)),:);
        X_train = X;
        X_train((1+j*l):min(l*(j+1), size(X,1)),:) = [];
        % find the covariance matrix 
        c = estim(cov(X_train), epsilons(i));
        % use the cv Variance as score
        dim = size(c,1);
        w = @(sigma) (pinv(sigma)*ones(dim, 1))/(ones(1, dim)*pinv(sigma)*ones(dim, 1));
        w_cv = w(c);
        returns = x_valid*w_cv;
        means(i) = means(i) + mean(returns);
    end
end
[argvalue, argmax] = max(means);
epsilon = epsilons(argmax);
COV = estim(cov(X), epsilon);
end

