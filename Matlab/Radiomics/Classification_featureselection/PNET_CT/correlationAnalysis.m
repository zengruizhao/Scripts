clear;clc;close all;
bestc = 199;besta = [];bestp = [177, 23, 1233, 252, 88];bestv = [];
load ../../Neuroendocrine/CT/new/Feature_5_3d_origin.mat
load ../../Neuroendocrine/CT/label_5_2cm.mat
data = feature_process(Feature, 1);
data1 = data(:, bestc);
data = feature_process(Feature, 2);
data = data(:, besta);
data1 = [data1, data];
data= feature_process(Feature, 3);
data = data(:, bestp);
data1 = [data1, data];
data= feature_process(Feature, 4);% V
data = data(:, bestv);
data1 = [data1, data];
label_ = label;
%
load ../../Neuroendocrine/CT/new/Feature_4_3d_origin.mat
load ../../Neuroendocrine/CT/label_4_2cm.mat
data = feature_process(Feature, 1);
data2 = data(:, bestc);
data = feature_process(Feature, 2);
data = data(:, besta);
data2 = [data2, data];
data= feature_process(Feature, 3);
data = data(:, bestp);
data2 = [data2, data];
data= feature_process(Feature, 4);
data = data(:, bestv);
data2 = [data2, data];
label_1 = label;
%%
data = [data1;data2];
label = [label_;label_1];
% features
G1 = data(logical(label(:, 1)==1), :);
G2 = data(logical(label(:, 1)==2), :);
label1 = label(logical(label(:, 1)==1), 1);
label2 = label(logical(label(:, 1)==2), 1);
data = [G1;G2];
data = zscore(data);
label = [label1;label2];
[r, p] = corr(data', 'type', 'Pearson');
image(r, 'CDataMapping', 'scaled');colorbar;
%%
figure
image(data, 'CDataMapping', 'scaled');
colorbar;
