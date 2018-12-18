function [a b] = ttestnew(datafile, trainlabel)

data = datafile;
ind1 = find(trainlabel ==1);
ind2 = find(trainlabel ==2);
controldata = data;
cancerdata = data;
controldata(ind1,:) = [];
cancerdata(ind2,:) = [];
lx = size(datafile,2);
for i=1:lx
    mean1(i) = mean(controldata(:,i));
    mean2(i) = mean(cancerdata(:,i));
    var1(i) = var(controldata(:,i));
    var2(i) = var(cancerdata(:,i));
end
tstat = (mean1 - mean2)./(sqrt(var1./length(ind2)+var2./length(ind1)));
t2 = abs(tstat);
t3 = t2./max(t2);
%plot(1:15154,t3,'.');
[a b] = sort(t2,'descend');