function [outputArg1] = bissection(f,interval, tol, maxit)
%BISSECTION Summary of this function goes here
%   Detailed explanation goes here
left = interval(1);
right = interval(2);
for i = 1:maxit
    if right-left<=2*tol
        break
    end
    m = (left+right)/2;
    if f(m)<0
        left = m;
    else
        right = m;
    end
    
end
outputArg1 = (left+right)/2;
end

