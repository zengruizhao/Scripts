clear;clc;
A = loadjson('ISIC_0000001_features.json');
data =cell2mat(struct2cell(A));