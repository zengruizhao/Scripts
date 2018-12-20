function [ c ] = CCC( var1, var2 )
%Concordance correlation coefficients(CCC) of two observers.
% Balagurunathan Y , Kumar V , Gu Y , et al. 
% Test¨CRetest Reproducibility Analysis of Lung CT Image Features[J]. 
% Journal of Digital Imaging, 2014, 27(6).
%range from -1:1
%   Detailed explanation goes here
    mu1 = mean(var1);
    mu2 = mean(var2);
    covarianceMatrix = cov(var1, var2);
    sigma1 = covarianceMatrix(1);
    sigma2 = covarianceMatrix(4);
    covariance = covarianceMatrix(2);
    c = 2*covariance/(sigma1^2 + sigma2^2 + (mu1-mu2)^2);
end

