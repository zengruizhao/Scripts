clear;clc;warning('off');
para.distrib = {'normal'};%kernel, normal, bad:mn, mvmn
para.prior = {'uniform'};%empirical, uniform
% best = [267, 1385, 266, 310, 379, 1369];% T1 origin *
best = [234, 312, 318, 397, 847, 1381];% T2 origin *
% best = [86,154,304,974,1304,884];% T2 dilate3 *
% best = [1369, 651, 1383, 327];% T1 dilate3
%%
% best = [140, 1385, 1369, 299, 302];% T1 dilate6
% best = [409, 636, 125, 405, 518];% T2 erode3
% best = [151, 86, 89, 152, 87];% T2 dilate3
% best = [379, 539, 216, 298, 9];% T2 dilate6
load /media/zzr/Data/git/Neuroendocrine/MRI/Feature_3d_origin_T2.mat
load /media/zzr/Data/git/Neuroendocrine/MRI/T2label_6_4.mat
data= feature_process(Feature);
data = data(:, best);%data = svm_scale(data);
datatraining = data(logical(~label(:, 3)), :);
datatesting = data(logical(label(:, 3)), :);%
%%
label2cm = label(logical(label(:, 3)), :);%
m = find(logical(label2cm(:, 4)));
%%
% data_testing_2cm = data(logical(label(:, 3))&logical(label(:, 4)), :);
labeltraining = label(logical(~label(:, 3)))-1;
labeltesting = label(logical(label(:, 3)))-1;%
label_testing_2cm = label(logical(label(:, 3))&logical(label(:, 4))) - 1;
[trainingset, ~]= svm_scale(datatraining);
[testingset, ~]= svm_scale(datatesting);
testingset2cm = testingset(m, :);
%
para.classifier_name = 'svmmine';%  'LDA' 'svmmine' 'ecoc' 'glm' 'kNN'% 'QDA' 'tree' 'NBayes' 'ensemble'
%%
[probability_pred, result, probability_pred_training]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset, testingset, labeltraining, labeltesting, para);
result.training.auc
result.testing.auc
%% plot ROC
fprTrain = result.training.FPR;
tprTrain = result.training.TPR;
fprTest = result.testing.FPR;
tprTest = result.testing.TPR;
plot(fprTrain, tprTrain);hold on;
plot(fprTest, tprTest);
%%
[probability_pred, result, probability_pred_training]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset, testingset2cm, labeltraining, label_testing_2cm, para);
result.testing.auc
fprTrain = result.training.FPR;
tprTrain = result.training.TPR;
fprTest = result.testing.FPR;
tprTest = result.testing.TPR;
plot(fprTest, tprTest);hold off;