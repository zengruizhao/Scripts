function [ icc ] = IIC( var1, var2 )
%Intraclass correlation coefficients
%   Detailed explanation goes here
    mu = mean([var1;var2]);
    sigma = std([var1;var2]);
    n = size(var1, 1);
    icc = sum((var1-mu).*(var2-mu))/((n-1)*sigma^2);

end

