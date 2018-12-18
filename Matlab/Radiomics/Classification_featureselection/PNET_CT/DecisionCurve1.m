clear;clc;warning('off');
addpath(genpath('../../Classification_featureselection'))
para.classifier_name = 'svmmine';%  'LDA' 'svmmine' 'ecoc' 'glm' 'kNN'% 'QDA' 'tree' 'NBayes' 'ensemble'
para.distrib = {'normal'};%kernel, normal, bad:mn, mvmn
para.prior = {'uniform'};%empirical, uniform
load ../../Neuroendocrine/CT/new/Feature_5_3d_origin.mat
load ../../Neuroendocrine/CT/label_5_2cm.mat
bestc = 199;
besta = [];
bestp = [177, 23, 1233, 252, 88];
bestv = [];
%%
data = feature_process(Feature, 1);
data1 = data(:, bestc);
data = feature_process(Feature, 5);
data = data(:, besta);
data1 = [data1, data];
data= feature_process(Feature, 3);
data = data(:, bestp);
data1 = [data1, data];
data= feature_process(Feature, 4);% V
data = data(:, bestv);
data1 = [data1, data];
label_ = label;
%%
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
datatraining = data(logical(~label(:, 3)), :);
datatesting = data(logical(label(:, 3)), :);
data2cm = data(logical(label(:, 4)), :);
labeltraining = label(logical(~label(:, 3)), 1) - 1;
labeltesting = label(logical(label(:, 3)), 1) - 1;
label2cm = label(logical(label(:, 4))) - 1;
%%
m = find(logical(label(:, 4)));
[trainingset, mean_value, std_value]= zscore(datatraining);
testingset = bsxfun(@minus,datatesting,mean_value);
testingset = bsxfun(@rdivide,testingset,std_value);
set2cm = bsxfun(@minus,data2cm,mean_value);
set2cm = bsxfun(@rdivide,set2cm,std_value);
allset = [trainingset;testingset];
alllabel = [labeltraining;labeltesting];
%%
[probability_pred, probability_probs] = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset, allset, labeltraining, alllabel, para);
[probability_pred_2cm, probability_probs_2cm] = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset, set2cm, labeltraining, label2cm, para);
Radiomics_testing = probability_pred;
Radiomics_2cm = probability_pred_2cm;

DC_testing(:, 1) = alllabel;
DC_testing(:, 2) = Radiomics_testing;
DC_testing(:, 3) = probability_probs;

DC_2cm(:, 1) = label2cm;
DC_2cm(:, 2) = Radiomics_2cm;
DC_2cm(:, 3) = probability_probs_2cm;

%
xlswrite('DC_testing1.xlsx', DC_testing);
xlswrite('DC_2cm1.xlsx', DC_2cm);
