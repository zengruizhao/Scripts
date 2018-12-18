function [a b] = ftestnew(datafile, trainlabel)

data = datafile;
ind1 = find(trainlabel ==1);
ind2 = find(trainlabel ==2);
controldata = data;
cancerdata = data;
controldata(ind1,:) = [];
cancerdata(ind2,:) = [];
h = controldata(1,:);
p = controldata(1,:);
lx = size(datafile,2);
parfor i=1:lx
    var1(i) = var(controldata(:,i));
    var2(i) = var(cancerdata(:,i));
end
Fstat = var1./var2;
for i=1:lx
    if (Fstat(i)<1)
        Fstat(i) = 1/Fstat(i);
    end
end
[a b] = sort(Fstat,'descend');