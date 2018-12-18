clear;clc;warning('off');
numofFeautre = 6;
para.classifier_name = 'svmmine';%  'LDA' 'svmmine' 'ecoc' 'glm' 'kNN'% 'QDA' 'tree' 'NBayes' 'ensemble'
para.distrib = {'normal'};%kernel, normal, bad:mn, mvmn
para.prior = {'uniform'};%empirical, uniform
best = [1385, 266, 310, 379, 1369];% T1 origin *
load E:\git\Neuroendocrine\MRI\Feature_3d_origin_T1.mat
load E:\git\Neuroendocrine\MRI\T1label_6_4.mat
data= feature_process(Feature);
data = data(:, best);
datatraining1 = data(logical(~label(:, 3)), :);
datatraining1(54, :) = [];%64 54
datatesting1 = data(logical(label(:, 3)), :);
%%
label2cm = label(logical(label(:, 3)), :);%
m = find(logical(label2cm(:, 4)));
best = [312, 318, 397, 847, 1381];
load E:\git\Neuroendocrine\MRI\Feature_3d_origin_T2.mat
load E:\git\Neuroendocrine\MRI\T2label_6_4.mat
data= feature_process(Feature);
data = data(:, best);
datatraining2 = data(logical(~label(:, 3)), :);
datatesting2 = data(logical(label(:, 3)), :);
datatraining = [datatraining1, datatraining2];
datatesting = [datatesting1, datatesting2];
labeltraining = label(logical(~label(:, 3)))-1;
labeltesting = label(logical(label(:, 3)))-1;
label_testing_2cm = label(logical(label(:, 3))&logical(label(:, 4))) - 1;
[trainingset, ~]= svm_scale(datatraining);
[testingset, ~]= svm_scale(datatesting);
testingset2cm = testingset(m, :);
%%
C = nchoosek(1:10, numofFeautre);
Result = zeros(size(C, 1), 2);
for i=1:size(C, 1)
    Training = trainingset(:, C(i, :));
    Testing = testingset(:, C(i, :));
    [~, result, ~]...
        = Lgenerate_Predicted_Label_of_test_data_no_FS(Training, Testing, labeltraining, labeltesting, para);
    Result(i, 1) = result.training.auc;
    Result(i, 2) = result.testing.auc;
end
[m, ~] = find(max(Result(:, 2))==Result);
bestFeature = C(m, :);
% max(Result(:, 2))
%%
[probability_pred, result, probability_pred_training]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset(:, C(m, :)), testingset(:, C(m, :)), labeltraining, labeltesting, para);
result.training.auc
result.testing.auc
%% plot ROC
fprTrain = result.training.FPR;
tprTrain = result.training.TPR;
fprTest = result.testing.FPR;
tprTest = result.testing.TPR;
plot(fprTrain, tprTrain);hold on;
plot(fprTest, tprTest);
[probability_pred, result, probability_pred_training]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset(:, C(m, :)), testingset2cm(:, C(m, :)), labeltraining, label_testing_2cm, para);
result.testing.auc
fprTest = result.testing.FPR;
tprTest = result.testing.TPR;
plot(fprTest, tprTest);hold off;
