%% preparation
clc; clear; close all;

%% sample 500 points
X = mvnrnd([0,0,0], diag([1,2,3]), 5);
%% ground truth and sample covariance matrix
disp('True Covariance Matrix:');
disp(diag([1,2,3]));
disp('Sample Covariance Matrix:');
disp(cov(X));
fprintf('MSE: %f\n', sum(sum((diag([1,2,3])-cov(X)).^2)));
%% benchmarks
[LW, RBLW, OAS] = benchmark_matrices(X);
disp('Ledoit Wolf Estimate:');
disp(LW);
fprintf('MSE: %f\n', sum(sum((diag([1,2,3])-LW).^2)));
disp('Rao-Blackwell Ledoit Wolf Estimate:');
disp(RBLW);
fprintf('MSE: %f\n', sum(sum((diag([1,2,3])-RBLW).^2)));
disp('Oracle Approximation Estimate:')
disp(OAS)
fprintf('MSE: %f\n', sum(sum((diag([1,2,3])-OAS).^2)));