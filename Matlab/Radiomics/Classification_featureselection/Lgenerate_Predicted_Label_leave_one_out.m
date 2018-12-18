%% generate predictive labels
% how to set the para?
% para.featureranking='mrmr';
% para.num_top_feature=15;
% para.classifier_name='LDA';
% para.T_on_predicted_pro=0.5;
% para.feature_selection_mode='one-shot'; or 'cross-validation'

function [labels_pred,probability_pred,result]=Lgenerate_Predicted_Label_leave_one_out(data_set,labels,para)
% addpath('C:\Nutstore\Nutstore\PathImAnalysis_Program\Program\LClassifier');
% addpath('C:\Nutstore\Nutstore\Repository\George-newbranch\george_code\newbranch\tools\FeatureSelectionTools');
% addpath(genpath('C:\Nutstore\Nutstore\Repository\George-newbranch\george_code\newbranch\tools\FeatureSelectionTools\MRMR\peng_toolbox\mRMR_0.9_compiled'));
% 
% addpath('/Users/chenglu/Nutstore/PathImAnalysis_Program/Program/LClassifier');
% addpath('/Users/chenglu/Nutstore/Repository/George-newbranch/george_code/newbranch/tools/FeatureSelectionTools');
% addpath(genpath('/Users/chenglu/Nutstore/Repository/George-newbranch/george_code/newbranch/tools/FeatureSelectionTools/MRMR/peng_toolbox/mRMR_0.9_compiled'));
% addpath('/Users/chenglu/Nutstore/Repository/George-newbranch/george_code/newbranch/tools/ClassificationPackage-2009/cross_validation');

alldataidx=1:size(data_set,1);
labels_pred=[];
probability_pred=[];

for i=1:size(data_set,1)
    tra_idx=setdiff(alldataidx,i);
    training_set = data_set(tra_idx,:);
    testing_set = data_set(i,:);
    training_labels = labels(tra_idx);
    testing_labels = labels(i);
    %% feature selection - one shot, we also can have CV to lock down the final features
    if strcmp(para.feature_selection_mode,'one-shot')
        [set_topfeatureindex, ~]= Featureselection_single_run(training_set,training_labels,para.feature_list,para);
    end
    
    if strcmp(para.feature_selection_mode,'cross-validation')
        para.set_classifier={para.classifier_name};
        para.set_featureselection={para.featureranking};
        [resultACC, resultAUC, result_feat_ranked, result_feat_scores, result_feat_idx_ranked]...
            =Leveluate_feat_using_diff_classifier_feature_selection(training_set,training_labels,para.feature_list,para);
        
        tmp = result_feat_idx_ranked{1};
        set_topfeatureindex = tmp(1:para.num_top_feature);
    end
    %% classification
    if strcmp(para.classifier_name,'NBayes')
        distrib = para.distrib;
        prior = para.prior;
        [temp_stats,methodstring] = Classify( para.classifier_name, training_set(:,set_topfeatureindex) , testing_set(:,set_topfeatureindex), training_labels(:), testing_labels(:), distrib, prior);
    else
        [temp_stats,methodstring] = Classify( para.classifier_name, training_set(:,set_topfeatureindex) , testing_set(:,set_topfeatureindex), training_labels(:), testing_labels(:));
    end
    probability_pred(i)=temp_stats.prediction;
    labels_pred(i) = logical(temp_stats.prediction>para.T_on_predicted_pro);
end

if size(labels_pred,1)~=size(labels,1)
    labels_pred=labels_pred';
end

result.recall=sum(labels_pred==1&labels==1)/sum(labels)*100;
result.specificity=sum(labels_pred==0&labels==0)/sum(~labels)*100;
result.accuracy=(sum(labels_pred==1&labels==1)+sum(labels_pred==0&labels==0))/length(labels)*100;
end
