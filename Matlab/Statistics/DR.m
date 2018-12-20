function [ dr ] = DR( var1, var2 )
%Dynamic range
% Balagurunathan Y , Kumar V , Gu Y , et al. 
% Test¨CRetest Reproducibility Analysis of Lung CT Image Features[J]. 
% Journal of Digital Imaging, 2014, 27(6).
%   Detailed explanation goes here
    dr = 1 - sum(abs(var1-var2))/(size(var1, 1)*(max([var1;var2]) - min([var1;var2])));

end

