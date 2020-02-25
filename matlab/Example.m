%% organization
clc; clear; close all;

%% KL
display('KL')
%preparation
sigma = [10 0 0; 0 2 0; 0 0 3]
epsilon = 1e-2;

% test correct functionality
sigma_star = KL_estimator(sigma, epsilon)
%pause;

% path of condition number
epsilons = linspace(0,2,1000);
epsilons = epsilons(2:end);
res = zeros(size(epsilons));
for ind=1:size(res,2)
    sigma_star = KL_estimator(sigma, epsilons(ind));
    res(ind) = cond(sigma_star);
end
figure;
plot(epsilons,res)
title('Condition Number Path (KL)');
xlabel('Epsilon');
ylabel('Condition Number');
%pause;

% path of eigenvalues
epsilons = linspace(0,2,1000);
epsilons = epsilons(2:end);
eigs = zeros(size(sigma,1),size(epsilons,2));
for ind=1:size(res,2)
    sigma_star = KL_estimator(sigma, epsilons(ind));
    [~,D] = eig(sigma_star);
    eigs(:,ind) = diag(D);
end
figure;
plot(epsilons,eigs)
title('Eigenvalue Path (KL)');
xlabel('Epsilon');
ylabel('Eigenvalues');

%pause;

%% Wasserstein
display('Wasserstein')

% preparation
sigma = [10 0 0; 0 2 0; 0 0 3]
epsilon = 1e-1;

% test correct functionality
sigma_star = WS_estimator(sigma, epsilon)
%pause;

% path of condition number
epsilons = linspace(0,5,1000);
epsilons = epsilons(2:end);
res = zeros(size(epsilons));
for ind=1:size(res,2)
    sigma_star = WS_estimator(sigma, epsilons(ind));
    res(ind) = cond(sigma_star);
end
figure;
plot(epsilons,res)
title('Condition Number Path (Wasserstein)');
xlabel('Epsilon');
ylabel('Condition Number');
%pause;

% path of eigenvalues
epsilons = linspace(0,5,1000);
epsilons = epsilons(2:end);
eigs = zeros(size(sigma,1),size(epsilons,2));
for ind=1:size(res,2)
    sigma_star = WS_estimator(sigma, epsilons(ind));
    [~,D] = eig(sigma_star);
    eigs(:,ind) = diag(D);
end
figure;
plot(epsilons,eigs)
title('Eigenvalue Path (Wasserstein)');
xlabel('Epsilon');
ylabel('Eigenvalues');

%pause;