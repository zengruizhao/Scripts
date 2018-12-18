function [m,s] = ttest(x1,x2)
% function [m,s] = ttest(x1,x2)
% x1 and x2 are data vectors
% the function returns their means m(1) and m(2) 
% and standard deviations s(1) and s(2)
% and give the result of a t-test for the
% difference between the means, using 
% an alpha level of 0.05
n1 = length(x1);  % size of sample 1
n2 = length(x2);  % size of sample 2
m(1) = mean(x1);
m(2) = mean(x2);
s(1) = std(x1);
s(2) = std(x2);
% pooled standard deviation
df = (n1+n2-2);  % pooled degrees of freedom
if df > 30
   error('Degrees of freedom > 30')
   end;
sp = sqrt(((n1-1)*s(1)^2 + (n2-1)*s(2)^2)/df);
% standard error of mean
se = sp*sqrt(1/n1 + 1/n2);
tc = (m(1) - m(2))/se % computed t
t5 = [12.7;4.30;3.18;2.78;2.57;2.45;2.37;2.31;2.26;2.23;...
      2.20;2.18;2.16;2.15;2.13;2.12;2.11;2.10;2.09;2.09;...
      2.08;2.07;2.07;2.06;2.06;2.06;2.05;2.05;2.05;2.04];
t = t5(df)  % theoretical t on null hypothesis
dif = t5(df) - abs(tc);
if dif < 0
   fprintf('\nNull Hypothesis rejected\n');
else
   fprintf('\nNull Hypothesis accepted\n')
end