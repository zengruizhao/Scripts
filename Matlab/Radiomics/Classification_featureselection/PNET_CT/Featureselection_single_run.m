function [set_topfeatureindex, set_topfeaturename]= Featureselection_single_run(data_set,data_labels,feature_list,para)
% function [set_topfeatureindex, set_topfeaturename,set_confident_score]= Featureselection_single_run(data_set,data_labels,feature_list,para)
%% using mrmr
if strcmp(para.featureranking,'mrmr')
    %         map the data in to binary values 0 1
    dataw_discrete=makeDataDiscrete_mrmr(data_set);
    %             dataw_discrete=data_set>t; check check check
    setAll=1:size(data_set,2);
    [set_topfeatureindex] = mrmr_mid_d(dataw_discrete(:,setAll), uint8(data_labels), para.num_top_feature);
end

%% using random forest
if strcmp(para.featureranking,'rf')
    options = statset('UseParallel','never','UseSubstreams','never');
    B = TreeBagger(50,data_set,data_labels,'FBoot',0.667, 'oobpred','on','OOBVarImp', 'on', 'Method','classification','NVarToSample','all','NPrint',4,'Options',options);
    variableimportance = B.OOBPermutedVarDeltaError;
    [t,idx]=sort(variableimportance,'descend');
    set_topfeatureindex=idx(1:para.num_top_feature);
end

if strcmp(para.featureranking,'ttest') | strcmp(para.featureranking,'wilcoxon')
    %% using ttest
    if strcmp(para.featureranking,'ttest')
        [TTidx,confidence] = prunefeatures_new(data_set, data_labels, 'ttestp');
        set_topfeatureindex=TTidx(confidence<0.05);
        if isempty(set_topfeatureindex)
            set_topfeatureindex=TTidx(1:min(para.num_top_feature*2,size(data_set,2)));
        end
    end
    
    if strcmp(para.featureranking,'wilcoxon')
        [TTidx,confidence] = prunefeatures_new(data_set, data_labels, 'wilcoxon');
        set_topfeatureindex=TTidx(confidence<0.3);
        if isempty(set_topfeatureindex)
            set_topfeatureindex=TTidx(1:min(para.num_top_feature*3,size(data_set,2)));
        end
    end
    
    %%% lock down top features with low correlation
    set_topfeatureindex=Lpick_top_n_features_with_pvalue_correlation(data_set,set_topfeatureindex,para.num_top_feature,para.correlation_factor);
end
set_topfeaturename=feature_list(set_topfeatureindex)';
end