function trainfeat=trainpercent(feature_rand, feature_cell,percent ,number)
% ���ٷֱȻ������������Ͷ�Ӧ��cell��������Ϊѵ���ĸ�����
trainfeat=[];
cellnum=floor((percent/100)*number); %cell������������
randnum=number-cellnum;  %����������
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