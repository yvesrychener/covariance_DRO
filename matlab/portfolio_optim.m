%% clear the workplace, add paths & load the data
% use "Average Value Weighted Returns -- Monthly"
clc; clear; close all;

addpath('data')
addpath('ours')
addpath('benchmarks')
addpath('benchmarks\lw')
addpath('benchmarks\oas_and_rblw')


load('data/asset_returns.mat')
load('data/returns_table.mat')

%% train-test split
train = asset_returns(1:144,:);
test = asset_returns(145:end,:);
%% calculate the covariance matrices
cov_sample = cov(train);
[cov_LW, cov_RBLW, cov_OAS] = benchmark_matrices(train);
%TODO ADD OURS WITH CV
f = @(c,e) WS_estimator(c,e);
cov_WS = variance_CV(f, train, logspace(-3,2, 100), 10);
f = @(c,e) KL_estimator(c,e);
cov_KL = variance_CV(f, train, logspace(-3,2, 100), 10);

%% compute the portfolio weights
dim = size(cov_sample,1);
w = @(sigma) (pinv(sigma)*ones(dim, 1))/(ones(1, dim)*pinv(sigma)*ones(dim, 1));

w_sample = w(cov_sample);
w_LW = w(cov_LW);
w_RBLW = w(cov_RBLW);
w_OAS = w(cov_OAS);
%TODO ADDO OURS
w_KL = w(cov_KL);
w_WS = w(cov_WS);

%% test the portfolio performance in the test set
r_sample = portfolio_performance(w_sample, test, 'Sample Covariance');
r_LW = portfolio_performance(w_LW, test, 'LW Covariance');
r_RBLW = portfolio_performance(w_RBLW, test, 'RBLW Covariance');
r_OAS = portfolio_performance(w_OAS, test, 'OAS Covariance');
r_WS = portfolio_performance(w_WS, test, 'WS Covariance');
r_KL = portfolio_performance(w_KL, test, 'KL Covariance');