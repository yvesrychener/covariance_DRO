%% preparation
clc; clear; close all;

%% definitions
n = 200;
d = cat(1, ones(20,1), 2*ones(40,1), 3*ones(40,1));
c = diag(d);
epsilons = logspace(-10,1, 100);
loss = @(X,Y) sqrt(mean((X-Y).^2, 'all'));
COV = 0;
LW = 0;
RBLW = 0;
OAS = 0;
WS = zeros(size(epsilons));
KL = zeros(size(epsilons));

for i=1:length(epsilons)
    r = simulations(c, loss, 1000, n, epsilons(i), epsilons(i));
    m = mean(r);
    COV = m(1);
    LW = m(2);
    RBLW = m(3);
    OAS = m(4);
    KL(i) = m(5);
    WS(i) = m(6);
end

%% plotting
semilogx(epsilons, COV*ones(size(epsilons)));
hold on;
semilogx(epsilons, LW*ones(size(epsilons)));
semilogx(epsilons, RBLW*ones(size(epsilons)));
semilogx(epsilons, OAS*ones(size(epsilons)));
semilogx(epsilons, WS);
semilogx(epsilons, KL);
legend('Sample Covariance', 'Ledoit-Wolf', 'Rao-Blackwell-Ledoit-Wolf', ...
    'Oracle Approximation', 'Wasserstein', 'Kullback-Leibler');
