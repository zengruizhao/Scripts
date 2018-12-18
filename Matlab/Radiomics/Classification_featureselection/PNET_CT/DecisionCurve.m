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
ki67 = label(:, 2);
datatraining = data(logical(~label(:, 3)), :);
datatesting = data(logical(label(:, 3)), :);
labeltraining = label(logical(~label(:, 3)), 1) - 1;
labeltesting = label(logical(label(:, 3)), 1) - 1;
labeltesting_2cm = label(logical(label(:, 3))&logical(label(:, 4))) - 1;
ki67training = ki67(logical(~label(:, 3)));
ki67testing = ki67(logical(label(:, 3)));
ki67_testing_2cm = ki67(logical(label(:, 3))&logical(label(:, 4)));
%%
label2cm = label(logical(label(:, 3)), :);%
m = find(logical(label2cm(:, 4)));
[trainingset, ~, mean_value, std_value]= svm_scale(datatraining);
testingset = bsxfun(@minus,datatesting,mean_value);
testingset = bsxfun(@rdivide,testingset,std_value);
testingset2cm = testingset(m, :);
%%
[probability_pred, result, probability_pred_training]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset, testingset, labeltraining, labeltesting, para);
[probability_pred_2cm, result2cm, ~]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset, testingset2cm, labeltraining, labeltesting_2cm, para);
Radiomics_training = probability_pred_training;
Radiomics_testing = probability_pred;
Radiomics_2cm = probability_pred_2cm;

DC_testing(:, 1) = labeltesting;
DC_testing(:, 2) = ki67testing;
DC_testing(:, 3) = Radiomics_testing;

DC_2cm(:, 1) = labeltesting_2cm;
DC_2cm(:, 2) = ki67_testing_2cm;
DC_2cm(:, 3) = probability_pred_2cm;
%
DC_training(:, 1) = labeltraining;
DC_training(:, 2) = ki67training;
DC_training(:, 3) = Radiomics_training;
xlswrite('DC_testing1.xlsx', DC_testing);
xlswrite('DC_2cm1.xlsx', DC_2cm);
xlswrite('DC_training1.xlsx', DC_training);
