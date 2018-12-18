clear;clc;warning('off');
addpath(genpath(pwd)); % add dependency path
%% 1**** try different classifier and feature combination in cross validation
para.intFolds=5;
para.intIter=20;
para.num_top_feature=6;
para.get_balance_sens_spec=0;
para.correlation_factor=0.9;
para.FSmethod = 'ttest';%FSmethod feature selection parameter 'wilcoxon'; 'bhattacharyya' 'roc'% better parameter: 'ttest' % bad: 'entropy' 'roc'
para.params.kernel = 'linear';% SVM parameters 'linear' % best parameter: 'linear';bad result:'rbf','sigmoid';low speed:'poly'
para.params.c_range = 0;
para.params.g_range = 0;
% % Nbayes parameters
para.distrib = {'normal'};%kernel, normal, bad:mn, mvmn
para.prior = {'uniform'};%empirical, uniform
%% temperory FS & classifier
para.set_classifier={'LDA','svm', 'glm', 'ecoc'};
para.set_featureselection={'wilcoxon','mrmr','rfe','relieff', 'ttest', 'FSmethod'};
%% pnet new
phase = 4;
load ../../Neuroendocrine/CT/new/Feature_5_3d_origin.mat
load ../../Neuroendocrine/CT/label_5.mat
phase_5 = phase;
if phase==2
   phase_5 = 5; %5
end
data= feature_process(Feature, phase_5);
data = data(logical(~label(:, 3)), :);
label_ = label(logical(~label(:, 3)));
load ../../Neuroendocrine/CT/new/Feature_4_3d_origin.mat
load ../../Neuroendocrine/CT/label_4.mat
data1 = feature_process(Feature, phase);
data1 = data1(logical(~label(:, 3)), :);
label_1 = label(logical(~label(:, 3)));
data = [data;data1];
label = [label_;label_1];
[data, indice]= svm_scale(data);
labels = label - 1;
%%
feature_list = ones(1,size(data,2));%dummy one if unavaliable % feature name
[structMaxAUC, resultACC, resultAUC, result_feat_ranked, result_feat_scores, result_feat_idx_ranked]...
    = Leveluate_feat_using_diff_classifier_feature_selection(data,labels,feature_list,para);
T = Lget_classifier_feature_slection_performance_table(resultACC, resultAUC, para);% look at T to see your performance !!