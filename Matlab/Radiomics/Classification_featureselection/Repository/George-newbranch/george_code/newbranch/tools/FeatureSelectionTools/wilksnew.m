function [a b] = wilksnew(datafile, trainlabel)
%   This file prints an array of the discriminatory features
%   datafile = 'ovarian_61902.data'
%   xvalues = 'ovarian_61902.names2.csv'

data = datafile;
ind1 = find(trainlabel ==1);
ind2 = find(trainlabel ==2);
controldata = data;
cancerdata = data;
controldata(ind1,:) = [];
cancerdata(ind2,:) = [];
Ti = zeros(1,size(datafile,2));
Wi = zeros(1,size(datafile,2));
lx = length(datafile(1,:));
for i=1:lx
    Ti(i) = sumsq(data(:,i));
    Wi(i) = sumsq(controldata(:,i)) + sumsq(cancerdata(:,i));
end
Lambda = Wi./Ti;
L = 1-Lambda;
%plot(1:15154,L,'g.');
[a b] = sort(Lambda);