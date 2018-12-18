clear;clc;warning('off');close all;
para.classifier_name = 'svmmine';%  'LDA' 'svmmine' 'ecoc' 'glm' 'kNN'% 'QDA' 'tree' 'NBayes' 'ensemble'
para.distrib = {'normal'};%kernel, normal, bad:mn, mvmn
para.prior = {'uniform'};%empirical, uniform
%%
% best = [88, 90]; % V origin
% best = [212, 35, 199, 266, 356];% C
% best = [177, 23, 1233, 252, 88];% P
% best = [101, 36, 37, 102, 3, 198, 93];% A2+5
% best = [19, 23, 33, 113];%A2+2
%%
phase = 4;
load ../../Neuroendocrine/CT/new/Feature_5_3d_origin.mat
load ../../Neuroendocrine/CT/label_5_2cm.mat
phase_5 = phase;
if phase==2
   phase_5 = 5; %5
end
data= feature_process(Feature, phase_5);
label_ = label;
load ../../Neuroendocrine/CT/new/Feature_4_3d_origin.mat
load ../../Neuroendocrine/CT/label_4_2cm.mat
data1 = feature_process(Feature, phase);
label_1 = label;
data = [data;data1];
label = [label_;label_1];
%%
data = data(:, best);
datatraining = data(logical(~label(:, 3)), :);
datatesting = data(logical(label(:, 3)), :);%
%% <2cm
label2cm = label(logical(label(:, 3)), :);%
m = find(logical(label2cm(:, 4)));
%%
labeltraining = label(logical(~label(:, 3)))-1;
labeltesting = label(logical(label(:, 3)))-1;%
label_testing_2cm = label(logical(label(:, 3))&logical(label(:, 4))) - 1;
[trainingset, ~, mean_value, std_value]= svm_scale(datatraining);
testingset = bsxfun(@minus,datatesting,mean_value);
testingset = bsxfun(@rdivide,testingset,std_value);
testingset2cm = testingset(m, :);
%%
[probability_pred_test, result, ~]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset, testingset, labeltraining, labeltesting, para);
%% plot ROC
fprTrain = result.training.FPR;
tprTrain = result.training.TPR;
fprTest = result.testing.FPR;
tprTest = result.testing.TPR;
plot(fprTrain, tprTrain, 'LineWidth', 2);hold on;
plot(fprTest, tprTest, 'LineWidth', 2);
%%
[probability_pred2cm, result2cm, probability_pred_training]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset, testingset2cm, labeltraining, label_testing_2cm, para);
fprTest = result2cm.testing.FPR;
tprTest = result2cm.testing.TPR;
plot(fprTest, tprTest, 'LineWidth', 2);hold off;
xlabel('1-Specificity', 'FontSize',16);
ylabel('Sensitivity', 'FontSize',16);
title('Receiver Operating Characteristic Curve')
legend(['Training(AUC ' num2str(roundn(result.training.auc, -3)) ')'], ...
    ['Testing(AUC ' num2str(roundn(result.testing.auc, -3)) ')'], ...
    ['Testing <2cm(AUC ' num2str(roundn(result2cm.testing.auc, -3)) ')']);
