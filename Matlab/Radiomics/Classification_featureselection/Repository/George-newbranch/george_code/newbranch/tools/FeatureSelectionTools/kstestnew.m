function [a b] = kstestnew(datafile,trainlabel)
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
k = controldata(1,:);
lx = size(datafile,2);
parfor i=1:lx
    [H P K] = kstest2(controldata(:,i),cancerdata(:,i));
    k(i) = K;
end
%plot(1:15154,k,'r.');
[a b] = sort(k,'descend');    