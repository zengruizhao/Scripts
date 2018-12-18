%%% 
% mRMR + Lasso -> LDA
%%%
close all; clear; clc;warning('off');
addpath(genpath('E:\git\Classification_featureselection/'));
phase = 4;
load ../../Neuroendocrine/CT/new/Feature_5_3d_origin.mat
load ../../Neuroendocrine/CT/label_5_2cm.mat
phase_5 = phase;
if phase==2
   phase_5 = 5; %5
end
data= feature_process(Feature, phase_5);
datatraining = data(logical(~label(:, 3)), :);
datatesting  = data(logical(label(:, 3)), :);
label_ = label(:, 1);
label_training = label_(logical(~label(:, 3)));
label_testing = label_(logical(label(:, 3)));
load ../../Neuroendocrine/CT/new/Feature_4_3d_origin.mat
load ../../Neuroendocrine/CT/label_4_2cm.mat
data1 = feature_process(Feature, phase);
data1training = data1(logical(~label(:, 3)), :);
data1testing = data1(logical(label(:, 3)), :);
label_1 = label(:, 1);
label_1training = label_1(logical(~label(:, 3)));
label_1testing = label_1(logical(label(:, 3)));
datatraining = [datatraining;data1training];
datatesting = [datatesting;data1testing];
labeltraining = [label_training;label_1training];
labeltesting = [label_testing;label_1testing];
[data, indice, mean_value, std_value]= svm_scale(datatraining);
labels = labeltraining - 1;
% test data
testingset = bsxfun(@minus,datatesting,mean_value);
testingset = bsxfun(@rdivide,testingset,std_value);
testinglabels = labeltesting - 1;

%  'baggedc45', 'onevsone_baggedc45', 
%  'baggedc45_notraining', 'baggedc45_notraining_multiclass',
%  'baggedc45_multiclass',
%  'nbayes', 'nbayes_notraining',
%  'svm', 'qda', 'lda'
shuffle = 1; Method = 'svmmine';
nIter = 20; n_fold = 5;
step_1 = 200; step_2 = ceil(size(data, 1) / 10);

feature_scores = [];
AUC = []; ACC = [];
TP = []; FP = []; TN =[]; FN = [];
parfor j=1:nIter
    [tra, tes]=GenerateSubsets('nFold', data, labels, shuffle, n_fold);
    decision=zeros(size(labels)); prediction=zeros(size(labels));
    for i=1:n_fold
        training_set = data(tra{i},:);
        testing_set = data(tes{i},:);
     
        training_labels = labels(tra{i});
        testing_labels = labels(tes{i});
        
        % mrmr 
        dataw_discrete = makeDataDiscrete_mrmr(training_set);
        setAll=1:size(training_set, 2);
        [idx_TTest] = mrmr_miq_d(dataw_discrete(:, setAll), ...
            training_labels, step_1);
        
        % lasso pick features after mrmr
%         lasso_tr_features = training_set;
%         lasso_te_features = testing_set;
        lasso_tr_features = training_set(:,idx_TTest);
        lasso_te_features = testing_set(:, idx_TTest);
        lasso_tr_labels = training_labels(:);
        [B, fitInfo] = lasso(lasso_tr_features, lasso_tr_labels, 'CV', 10);
        auc = [];
        best_idx = {}; temp_feature_scores = double(zeros(1, size(data, 2)));
        LAMBDA = [fitInfo.IndexMinMSE, fitInfo.Index1SE];
        for oneLambda = min(LAMBDA):max(LAMBDA)%1:size(fitInfo.Lambda, 2)
            if oneLambda==min(LAMBDA)||oneLambda==max(LAMBDA)
                selected = B(:, oneLambda);
                selected_non_zero_idx = find(selected ~= 0);
                if length(selected_non_zero_idx) > step_2
                    [~, selected_non_zero_idx_best] = sort(selected);
                    selected_non_zero_idx_best = selected_non_zero_idx_best(1:step_2);
                else
                    selected_non_zero_idx_best = selected_non_zero_idx;
                end
                if isempty(selected_non_zero_idx)
                    continue;
                end
                best_idx{oneLambda} = selected_non_zero_idx_best;

                % LDA
                trf = lasso_tr_features(:, selected_non_zero_idx_best);
                tef = lasso_te_features(:, selected_non_zero_idx_best);

                [temp_stats, methodstring] = Classify( Method, trf, tef, ...
                    training_labels(:), testing_labels(:));
                [FPR,TPR,T,auc(oneLambda),OPTROCPT,~,~] = perfcurve(testing_labels,temp_stats.prediction,1);
            end
        end
        %%%%%%
        lamdba_auc_idx_best = find(auc == max(auc));
        temp_best_feature_idx = [];
        if length(lamdba_auc_idx_best) > 1
            for m = 1:length(lamdba_auc_idx_best)
                temp_best_feature_idx = ...%[temp_best_feature_idx; best_idx{lamdba_auc_idx_best(m)}];
                    [temp_best_feature_idx idx_TTest(best_idx{lamdba_auc_idx_best(m)})];
            end
            temp_best_feature_idx = unique(temp_best_feature_idx);
        end
        temp_feature_scores(temp_best_feature_idx) = 1;
        feature_scores = [feature_scores; temp_feature_scores]; 
    end
end
[~, final_selected_idx_sort] = sort(sum(feature_scores));
final_selected_idx = final_selected_idx_sort(end - step_2:end);
figure, hold on;
% on Training set
[resulttraining, ~] = Classify( Method, ...
    data(:, final_selected_idx), data(:, final_selected_idx), ...
    labels(:), labels(:));
[resulttesting, ~] = Classify( Method, ...
    data(:, final_selected_idx), testingset(:, final_selected_idx), ...
    labels(:), testinglabels(:));
[X1, Y1, T, train_auc] = perfcurve(labels', resulttraining.prediction', 1);
[X2, Y2, T, test_auc] = perfcurve(testinglabels', resulttesting.prediction', 1);
plot(X1, Y1, 'linewidth', 2);
plot(X2, Y2, 'linewidth', 2);
legend(['trainset: ' num2str(train_auc)], ['testset: ' num2str(test_auc)]);
