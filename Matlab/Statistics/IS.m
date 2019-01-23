function [ is ] = IS( var1, var2, N )
% Preparation-Induced Instability
% applied to 2 datasets
% var1 and var2: same as ranksum
% N:the number of iters
% Detailed explanation goes here
    U = [];
    while N
        n1 = var1(randperm(size(var1, 1), round(size(var1, 1) / 2)), :);
        n2 = var2(randperm(size(var2, 1), round(size(var2, 1) / 2)), :);
        p = ranksum(n1, n2);
        if p<0.05,U(end+1)=1;else,U(end+1)=0;end
        N=N-1;
    end
    is = sum(U) / length(U);
end