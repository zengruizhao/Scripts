clear;clc;warning('off');
%% 1**** try different classifier and feature combination in cross validation
para.intFolds=3;
para.intIter=20;
para.num_top_feature=5;
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
% para.set_classifier={'svm'};
% para.set_featureselection={'FSmethod'};
%% best FS & classifier for Pancrease
para.set_classifier={'LDA','svm', 'glm', 'ecoc'};
% para.set_featureselection={'rfe'};
%% experiment all 
% para.set_classifier={'LDA','svm','NBayes','QDA', 'ensemble', 'glm', 'ecoc'};% bad performance:'tree' , 'knn'
para.set_featureselection={'wilcoxon','mrmr','rfe','relieff', 'ttest', 'FSmethod'};
%% all the ways of FS & classifiers
% para.set_classifier={'LDA','svm','NBayes'};% can't use : 'ensemble';useless:knn;Bad:'BaggedC45';error:QDA
% para.set_featureselection={'wilcoxon','mrmr','rfe','relieff','l0'};
% add 'inffs','ecfs','relieff','mutinffs','fsv','laplacian','mcfs','rfe','l0','fisher','udfs','llcfs','cfs' later;
% better:'relieff','rfe','l0';% slow: 'ecfs', 'udfs','rf'; % can't use:
% 'mutinffs', 'fsv', 'fisher' because of the NaN;difficult to use
% :'ttest',FSmethod,inffs,cfs;error:'llcfs','l0','mcfs';Bad:'laplacian','llcfs','mcfs'
%%
load E:\git\Neuroendocrine\MRI\Feature_3d_origin_T2.mat
load E:\git\Neuroendocrine\MRI\T2label_6_4.mat
data= feature_process(Feature);
data = data(logical(~label(:, 3)), :);
label = label(logical(~label(:, 3)));
label = label(:, 1);
[data, indice]= svm_scale(data);
labels = label - 1;
%%
feature_list = ones(1,size(data,2));%dummy one if unavaliable % feature name
[structMaxAUC, resultACC, resultAUC, result_feat_ranked, result_feat_scores, result_feat_idx_ranked]...
    = Leveluate_feat_using_diff_classifier_feature_selection(data,labels,feature_list,para);
T = Lget_classifier_feature_slection_performance_table(resultACC, resultAUC, para);% look at T to see your performance !!
