clear;clc;warning('off');
para.classifier_name = 'svmmine';%  'LDA' 'svmmine' 'ecoc' 'glm' 'kNN'% 'QDA' 'tree' 'NBayes' 'ensemble'
para.distrib = {'normal'};%kernel, normal, bad:mn, mvmn
para.prior = {'uniform'};%empirical, uniform
best = [310, 379];% T1 origin *
load ../../Neuroendocrine/MRI/Feature_3d_origin_T1.mat
load ../../Neuroendocrine/MRI/T1label_6_4.mat
data= feature_process(Feature);
data = data(:, best);
datatraining1 = data(logical(~label(:, 3)), :);
datatraining1(54, :) = [];%64 54
datatesting1 = data(logical(label(:, 3)), :);
%%
label2cm = label(logical(label(:, 3)), :);%
m = find(logical(label2cm(:, 4)));
best = [312, 397, 847, 1381];
load ../../Neuroendocrine/MRI/Feature_3d_origin_T2.mat
load ../../Neuroendocrine/MRI/T2label_6_4.mat
ki67 = label(:, 2);
data= feature_process(Feature);
data = data(:, best);
datatraining2 = data(logical(~label(:, 3)), :);
datatesting2 = data(logical(label(:, 3)), :);
datatraining = [datatraining1, datatraining2];
datatesting = [datatesting1, datatesting2];
labeltraining = label(logical(~label(:, 3)))-1;
ki67training = ki67(logical(~label(:, 3)));
labeltesting = label(logical(label(:, 3)))-1;
ki67testing = ki67(logical(label(:, 3)));
label_testing_2cm = label(logical(label(:, 3))&logical(label(:, 4))) - 1;
ki67_testing_2cm = ki67(logical(label(:, 3))&logical(label(:, 4)));
[trainingset, ~]= svm_scale(datatraining);
[testingset, ~]= svm_scale(datatesting);
testingset2cm = testingset(m, :);
%%
[probability_pred, result, probability_pred_training]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset, testingset, labeltraining, labeltesting, para);
[probability_pred_2cm, result2cm, ~]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset, testingset2cm, labeltraining, label_testing_2cm, para);
Radiomics_training = probability_pred_training;
Radiomics_testing = probability_pred;
Radiomics_2cm = probability_pred_2cm;

DC_testing(:, 1) = labeltesting;
DC_testing(:, 2) = ki67testing;
DC_testing(:, 3) = Radiomics_testing;

DC_2cm(:, 1) = label_testing_2cm;
DC_2cm(:, 2) = ki67_testing_2cm;
DC_2cm(:, 3) = probability_pred_2cm;
%
DC_training(:, 1) = labeltraining;
DC_training(:, 2) = ki67training;
DC_training(:, 3) = Radiomics_training;
% xlswrite('DC_testing.xlsx', DC_testing);
% xlswrite('DC_2cm.xlsx', DC_2cm);
% xlswrite('DC_training.xlsx', DC_training);