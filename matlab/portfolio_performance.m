function [s,m, sharpe] = portfolio_performance(w, X, print_name)
%PORTFOLIO_PERFORMANCE Calculate performance for portfolio weights
%   w : portfolio weights
%   X : Asset (excess) returns
%   print_name : name of the method if printing is desired, else boolean
%   False
%   Returns : s, m, sharpe : stdev, mean, sharpe of returns
returns = X*w;
s = std(returns);
m = mean(returns);
sharpe = m/s;

if print_name
    fprintf('Results for %s:\n', print_name)
    fprintf('Return Mean :\t%f\n', m);
    fprintf('Return STD :\t%f\n', s);
    fprintf('Return Sharpe :\t%f\n', sharpe);
    fprintf('\n\n')
end

end

