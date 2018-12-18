function [value] = sumsq(X)

m = mean(X);
x = X-m;

value = 0;
for i = 1:length(X)
    value = value + x(i)*x(i);
end
