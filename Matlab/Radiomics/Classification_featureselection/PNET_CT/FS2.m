clear;clc;warning('off');
numofFeautre = 5;
para.distrib = {'normal'};%kernel, normal, bad:mn, mvmn
para.prior = {'uniform'};%empirical, uniform
para.classifier_name = 'svmmine';%  'svmmine' 'ecoc' 'glm', 'QDA'% 'tree' 'NBayes' 'ensemble'
% best = [23, 241, 88, 176, 90, 156, 206, 100, 93, 148, 269, 315];% V origin
best = [138, 144, 266, 212 239, 251, 35, 174, 199, 356, 1076, 1226];% C
% best = [177, 23, 1233, 241, 161, 1249, 252, 88, 161, 246, 187, 219, 223, 1231];% P
% best = [101, 36, 37, 102, 3, 198, 23, 93, 88, 113, 98, 18, 19, 1270, 209, 1245];% A2_5
% best = [223, 88, 3, 4, 101, 19, 263, 1270, 239, 36, 93, 23, 113, 33, 232];% A2_2
%%
phase = 1;
load ../../Neuroendocrine/CT/new/Feature_5_3d_origin.mat
load ../../Neuroendocrine/CT/label_5_2cm.mat
phase_5 = phase;
if phase==2
   phase_5 = 2; %5
end
data= feature_process(Feature, phase_5);
label_ = label;
load ../../Neuroendocrine/CT/new/Feature_4_3d_origin.mat
load ../../Neuroendocrine/CT/label_4_2cm.mat
data1 = feature_process(Feature, phase);
label_1 = label;
Data = [data;data1];
label = [label_;label_1];
%%
C = nchoosek(best, numofFeautre);
Result = zeros(size(C, 1), 2);
for i=1:size(C, 1)
    fprintf('%d \\ %d\n', i, size(C, 1));
    data = Data(:, C(i, :));
    datatraining = data(logical(~label(:, 3)), :);
    datatesting = data(logical(label(:, 3)), :);
    labeltraining = label(logical(~label(:, 3)))-1;
    labeltesting = label(logical(label(:, 3)))-1;
    [trainingset, ~, mean_value, std_value]= svm_scale(datatraining);
    testingset = bsxfun(@minus,datatesting,mean_value);
    testingset = bsxfun(@rdivide,testingset,std_value);
    [probability_pred, result, probability_pred_training]...
        = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset, testingset, labeltraining, labeltesting, para);
    Result(i, 1) = result.training.auc;
    Result(i, 2) = result.testing.auc;
    if result.training.auc < result.testing.auc; Result(i, 1)=0;Result(i, 2)=0;end
end
[m, ~] = find(max(Result(:, 2))==Result);
bestFeature = C(m, :);
fprintf('6_4 max auc: %f\n', max(Result(:, 2)));


