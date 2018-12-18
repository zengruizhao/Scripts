clear;clc;warning('off');close all;
numofFeautre = 6;
para.classifier_name = 'svmmine';%  'LDA' 'svmmine' 'ecoc' 'glm' 'kNN'% 'QDA' 'tree' 'NBayes' 'ensemble'
para.distrib = {'normal'};%kernel, normal, bad:mn, mvmn
para.prior = {'uniform'};%empirical, uniform
%%
% bestc = [212, 35, 199, 266, 356];
% % besta = [101, 36, 37, 102, 3, 198, 93];%2+5
% besta = [19, 23, 33, 113];%2+2
% bestp = [177, 23, 1233, 252, 88];
% bestv = [88, 90];
%%
bestc = 199;
besta = [];
bestp = [177, 23, 1233, 252, 88];
bestv = [];
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
labeltraining = label(logical(~label(:, 3)), 1) - 1;
labeltesting = label(logical(label(:, 3)), 1) - 1;
labeltesting_2cm = label(logical(label(:, 3))&logical(label(:, 4))) - 1;
%%
label2cm = label(logical(label(:, 3)), :);%
m = find(logical(label2cm(:, 4)));
[trainingset, ~, mean_value, std_value]= svm_scale(datatraining);
testingset = bsxfun(@minus,datatesting,mean_value);
testingset = bsxfun(@rdivide,testingset,std_value);
testingset2cm = testingset(m, :);
%%
C = nchoosek(1:size(testingset, 2), numofFeautre);
Result = zeros(size(C, 1), 2);
for i=1:size(C, 1)
    fprintf('%d \\ %d\n', i, size(C, 1));
    Training = trainingset(:, C(i, :));
    Testing = testingset(:, C(i, :));
    [~, result, ~]...
        = Lgenerate_Predicted_Label_of_test_data_no_FS(Training, Testing, labeltraining, labeltesting, para);
    Result(i, 1) = result.training.auc;
    Result(i, 2) = result.testing.auc;
    if result.training.auc < result.testing.auc; Result(i, 1)=0;Result(i, 2)=0;end
end
[m, ~] = find(max(Result(:, 2))==Result(:, 2));
bestFeature = C(m, :);
idx_best =1;
%%
[probability_pred_test, result, ~]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset(:, C(m(idx_best), :)), ...
    testingset(:, C(m(idx_best), :)), labeltraining, labeltesting, para);
%% plot ROC
fprTrain = result.training.FPR;
tprTrain = result.training.TPR;
fprTest = result.testing.FPR;
tprTest = result.testing.TPR;
plot(fprTrain, tprTrain, 'LineWidth', 2);hold on;
plot(fprTest, tprTest, 'LineWidth', 2);
[probability_pred_2cm, result2cm, probability_pred_training]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset(:, C(m(idx_best), :)), ...
    testingset2cm(:, C(m(idx_best), :)), labeltraining, labeltesting_2cm, para);
fprTest = result2cm.testing.FPR;
tprTest = result2cm.testing.TPR;
plot(fprTest, tprTest, 'LineWidth', 2);hold off;
xlabel('1-Specificity', 'FontSize',16);
ylabel('Sensitivity', 'FontSize',16);
title('Receiver Operating Characteristic Curve')
legend(['Training(AUC ' num2str(roundn(result.training.auc, -3)) ')'], ...
    ['Testing(AUC ' num2str(roundn(result.testing.auc, -3)) ')'], ...
    ['Testing <2cm(AUC ' num2str(roundn(result2cm.testing.auc, -3)) ')']);
