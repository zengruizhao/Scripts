clear;clc;warning('off');
addpath(genpath(pwd)); % add dependency path
%% 1**** try different classifier and feature combination in cross validation
para.intFolds=3;
para.intIter=1;
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
% para.set_classifier={'NBayes'};
% para.set_featureselection={'wilcoxon','rfe'};
%% best FS & classifier for Pancrease
para.set_classifier={'svm'};
% para.set_featureselection={'llcfs'};
% para.set_classifier={'svm', 'glm', 'ecoc','LDA'};% 'LDA',
para.set_featureselection={'wilcoxon','mrmr','rfe','relieff', 'ttest', 'FSmethod'};
%% experiment all 
% para.set_classifier={'LDA','svm','NBayes','BaggedC45'};% can't use : 'ensemble';useless:knn;Bad:'BaggedC45';error:QDA
% para.set_featureselection={'wilcoxon','mrmr','rfe','relieff','l0','laplacian','llcfs','mcfs'};
%% all the ways of FS & classifiers
% para.set_classifier={'LDA','svm','NBayes'};% can't use : 'ensemble';useless:knn;Bad:'BaggedC45';error:QDA
% para.set_featureselection={'wilcoxon','mrmr','rfe','relieff','l0'};
% add 'inffs','ecfs','relieff','mutinffs','fsv','laplacian','mcfs','rfe','l0','fisher','udfs','llcfs','cfs' later;
% better:'relieff','rfe','l0';% slow: 'ecfs', 'udfs','rf'; % can't use:
% 'mutinffs', 'fsv', 'fisher' because of the NaN;difficult to use
% :'ttest',FSmethod,inffs,cfs;error:'llcfs','l0','mcfs';Bad:'laplacian','llcfs','mcfs'
%%
% -data should be a m x n matrix, in which m is the patient number and n is
% the feature number
% -labels is a vector to indicate the outcome, normally set to 0 or 1
% labels = classes;
%% best
% phase = 1;
% load ../Neuroendocrine/Feature_5_erode_3.mat
% load ../Neuroendocrine/label_5_2cm.mat
% % kinetics_feature = kinetics(Feature);
% phase_5 = phase;
% if phase==2
%    phase_5 = 5; %5
% end
% [data, delete]= feature_process(Feature, phase_5);
% data = data(label~=0, :);
% label = label(label~=0, :);
% label_ = label;
% % data(delete, :) = [];
% % label_(delete, :) = [];
% load ../Neuroendocrine/Feature_4_erode_3.mat
% load ../Neuroendocrine/label_4_2cm.mat
% data1 = feature_process(Feature, phase);
% data1 = data1(label~=0, :);
% data = [data; data1];
% label = [label_; label(label~=0, :)];
% [data indice]= svm_scale(data);
% labels = label - 1;
%% pnet new
phase = 2;
load ../../Neuroendocrine/CT/new/Feature_5_3d_origin.mat
load ../../Neuroendocrine/CT/label_5_2cm.mat
phase_5 = phase;
if phase==2
   phase_5 = 2; %5
end
data= feature_process(Feature, phase_5);
data = data(logical(~label(:, 3)), :);
label_ = label(:, 1);
label_ = label_(logical(~label(:, 3)));
load ../../Neuroendocrine/CT/new/Feature_4_3d_origin.mat
load ../../Neuroendocrine/CT/label_4_2cm.mat
data1 = feature_process(Feature, phase);
data1 = data1(logical(~label(:, 3)), :);
label_1 = label(:, 1);
label_1 = label_1(logical(~label(:, 3)));
data = [data;data1];
label = [label_;label_1];
[data, indice]= svm_scale(data);
labels = label - 1;
%%
feature_list = ones(1,size(data,2));%dummy one if unavaliable % feature name
[structMaxAUC, resultACC, resultAUC, result_feat_ranked, result_feat_scores, result_feat_idx_ranked]...
    = Leveluate_feat_using_diff_classifier_feature_selection(data,labels,feature_list,para);
T = Lget_classifier_feature_slection_performance_table(resultACC, resultAUC, para);% look at T to see your performance !!
%% 2**** if you want to use leave one out to get the predicted labels, use the code below
% para.featureranking='wilcoxon';
% para.num_top_feature=3;
% para.classifier_name='LDA';
% para.T_on_predicted_pro=0.5;
% para.feature_selection_mode='cross-validation';%'one-shot'; %
% para.feature_list=feature_list;
% [labels_pred,probability_pred,result] = Lgenerate_Predicted_Label_leave_one_out(data,labels,para);
%% 3****build classifiers using selected features from training set
% load '/media/zzr/Data/git/Intracranial Atherosclerotic Plaque/output/features_extracted/Preprocess/SVM_DATA';
% training_set = svm_feature{1, 1}{1, 1};
% training_labels = svm_label{1, 1}{1, 1}';
% testing_set = svm_feature{1, 1}{1, 1};
% testing_labels = svm_label{1, 1}{1, 1}';
% [training_set indice_training] = svm_scale(training_set);
% testing_set(:,indice_training) = [];
% testing_set  = svm_scale_no_clear(testing_set);
% feature_list = ones(1,size(training_set,2));
% %
% para.intFolds=5;
% para.intIter=10;
% para.num_top_feature=2;
% para.get_balance_sens_spec=0;
% para.featureranking='wilcoxon';
% para.classifier_name='NBayes';
% para.T_on_predicted_pro=0.5;
% para.feature_selection_mode='cross-validation';%'one-shot'; %
% para.feature_list=feature_list;
% para.distrib = {'normal'};%kernel, normal, bad:mn, mvmn
% para.prior = {'uniform'};%empirical, uniform
% %%
% [labels_pred,probability_pred,result]=Lgenerate_Predicted_Label_of_test_data(training_set,testing_set,training_labels, testing_labels,para);
% %%% build the final classifer using QDA
% 
% T_predict=0.5; % threshold on the predicted probability obtain by the classifier
% 
% % data_validation - is the data from validation set, in which the feature
% % dimension is the same as the the training data (only keep the slected features)
% % labels - is a vector, with 0 and 1
% try 
%     [~,~,probs,~,c] = classify(data_validation,data_train,labels,'quadratic'); 
% catch err
%     [~,~,probs,~,c] = classify(data_validation,data_train,labels,'diagquadratic'); 
% end
% label_qda=zeros(size(data_validation,1),1);
% label_qda(probs(:,2)>T_predict)=1;
% % sum(label_qda)
% 
% 
% %%% build the final classifer using LDA
% try 
%     [~,~,probs,~,c] = classify(data_validation,data_train,labels,'linear'); 
% catch err
%     [~,~,probs,~,c] = classify(data_validation,data_train,labels,'diaglinear'); 
% end
% 
% label_lda=zeros(size(data_validation,1),1);
% label_lda(probs(:,2)>T_predict)=1;
% % sum(label_lda)
% 
% 
%%% build the final classifer using RF
% data_train=good_featuregroup_data;
% data_validation=good_featuregroup_data;
% methodstring = 'BaggedC45';
% options = statset('UseParallel','never','UseSubstreams','never');
% C_rf = TreeBagger(50,data_train,labels,'FBoot',0.667,'oobpred','on','Method','classification','NVarToSample','all','NPrint',4,'Options',options);    % create bagged d-tree classifiers from training
% 
% [Yfit,Scores] = predict(C_rf,data_validation);   % use to classify testing
% % [Yfit,Scores] = predict(C_rf,data_train);   % use to classify testing
% label_lda=str2double(Yfit);
% % sum(label_lda)
% % accuracy_RF_reuse=sum(label_lda==labels')/length(labels);

% %% 4**** check the classification perfromance in test set
% [recall,specificity,accuracy]=Lcal_recall_spe_acc(groundtruth_label,predicted_label);