function [ cv ] = CV( var1, var2 )
%coefficient of variance
%   Detailed explanation goes here
    mu = [mean(var1), mean(var2)];
    covariance = cov(var1, var2);
    cv = sqrt(mu*covariance*mu'/(mu*mu')^2);
end

