function trainfeat=trainpercent(feature_rand, feature_cell,percent ,number)
% 按百分比混合随机块特征和对应的cell负样本作为训练的负样本
trainfeat=[];
cellnum=floor((percent/100)*number); %cell负样本的数量
randnum=number-cellnum;  %随机块的数量
[randsize x1]=size(feature_rand);
[cellsize x2]=size(feature_cell);
for i=1:randnum
    feat=feature_rand(randi([1 randsize]),:);
    trainfeat=[trainfeat; feat];
end
for i=1:cellnum
    feat=feature_cell(randi([1 cellsize]),:);
    trainfeat=[trainfeat; feat];
end