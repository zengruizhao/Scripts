function [a b] = Wilkcoxnew(datafile, trainlabel)
%   This file prints an array of the discriminatory features
%   datafile = 'ovarian_61902.data'
%   xvalues = 'ovarian_61902.names2.csv'

data = datafile;
ind1 = find(trainlabel ==0);
ind2 = find(trainlabel ==1);
controldata = data;
cancerdata = data;
controldata(ind1,:) = [];
cancerdata(ind2,:) = [];
lx = size(datafile,2);
s1 = zeros(1,lx);
for i=1:lx
    [P,H, stat] = ranksum(controldata(:,i),cancerdata(:,i));
    [P,H, stat2] = ranksum(cancerdata(:,i),controldata(:,i));
    s1(i) = min(stat.ranksum,stat2.ranksum);
    %s1(i) = P.*-1;
    s1(i) = P;
end
[a b] = sort(s1,'ascend');
%[a b] = sort(s1,'descend');