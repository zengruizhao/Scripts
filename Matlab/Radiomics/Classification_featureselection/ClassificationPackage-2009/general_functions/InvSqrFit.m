function [Y2] = InvSqrFit(X,Y,X2)

options.TolFun = 1e-12;
options.TolCon = 1e-12;
options.MaxIter = 1000;
options.MaxFunEvals = 1000;

params = fmincon(@(x) myFunc(x,X,Y), [0,0,0],[],[],[],[],[0 0 0],[Inf Inf Inf],[],options);
% params = fminsearch(@(x) myFunc(x,X,Y),[1,1,1]);
Y2 = params(1)*X2.^(-params(2)) + params(3);

function f = myFunc(x,n,err)
% x(1): a, x(2): alpha, x(3): b
f = sum((x(1) * n.^(-x(2)) + x(3) - err).^2);