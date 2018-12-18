clear;clc;warning('off');
numofFeautre = 6;
para.distrib = {'normal'};%kernel, normal, bad:mn, mvmn
para.prior = {'uniform'};%empirical, uniform
para.classifier_name = 'svmmine';%  'LDA' 'svmmine' 'ecoc' 'glm' 'kNN'% 'QDA' 'tree' 'NBayes' 'ensemble'
best = [312, 405, 539, 420, 234, 518, 815, 1381, 847, 1259, 443, 318, 410, 638, 1174, 770, 425, 656, 397, 536, 306, 268, 530];% T2 origin
% best = [267, 266, 1369, 1384, 1234, 1385, 337, 310, 379];%T1 origin
% best = [1369, 337, 1385, 324, 351, 651, 1383, 1370, 327, 1384];%T1 dilate3
% best = [86, 154, 304, 974, 1259, 1304, 539, 884, 584];%T2 dilate3
load E:\git\Neuroendocrine\MRI\Feature_3d_origin_T2.mat
load E:\git\Neuroendocrine\MRI\T2label_6_4.mat
Data= feature_process(Feature);
C = nchoosek(best, numofFeautre);
Result = zeros(size(C, 1), 2);
for i=1:size(C, 1)
    data = Data(:, C(i, :));
    datatraining = data(logical(~label(:, 3)), :);
    datatesting = data(logical(label(:, 3)), :);%
    data_testing_2cm = data(logical(label(:, 3))&logical(label(:, 4)), :);
    labeltraining = label(logical(~label(:, 3)))-1;
    labeltesting = label(logical(label(:, 3)))-1;%
    trainingset= svm_scale(datatraining);
    testingset= svm_scale(datatesting);
    [probability_pred, result, probability_pred_training]...
        = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset, testingset, labeltraining, labeltesting, para);
    Result(i, 1) = result.training.auc;
    Result(i, 2) = result.testing.auc;
end
[m, ~] = find(max(Result(:, 2))==Result);
bestFeature = C(m, :);
fprintf('6_4 max auc: %f\n', max(Result(:, 2)));
best = unique(bestFeature);
C = nchoosek(best, numofFeautre);
Result = zeros(size(C, 1), 2);
load E:\git\Neuroendocrine\MRI\T2label_7_3.mat
for i=1:size(C, 1)
    data = Data(:, C(i, :));
    datatraining = data(logical(~label(:, 3)), :);
    datatesting = data(logical(label(:, 3)), :);%
    data_testing_2cm = data(logical(label(:, 3))&logical(label(:, 4)), :);
    labeltraining = label(logical(~label(:, 3)))-1;
    labeltesting = label(logical(label(:, 3)))-1;%
    trainingset= svm_scale(datatraining);
    testingset= svm_scale(datatesting);
    [probability_pred, result, probability_pred_training]...
        = Lgenerate_Predicted_Label_of_test_data_no_FS(trainingset, testingset, labeltraining, labeltesting, para);
    Result(i, 1) = result.training.auc;
    Result(i, 2) = result.testing.auc;
end
[m, ~] = find(max(Result(:, 2))==Result);
bestFeature = C(m, :);
fprintf('7_3 max auc: %f\n', max(Result(:, 2)));


