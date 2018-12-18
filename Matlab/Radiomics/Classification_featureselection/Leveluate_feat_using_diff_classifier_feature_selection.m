function [structMaxAUC, resultACC,resultAUC,result_feat_ranked,result_feat_scores,result_feat_idx_ranked]=...
    Leveluate_feat_using_diff_classifier_feature_selection(data_all,labels,feature_list_all,para)

if ~isfield(para,'set_classifier')
    set_classifier={'LDA','QDA','BaggedC45'};
    set_featureselection={'wilcoxon','mrmr','rf','ttest'};
else
    set_classifier=para.set_classifier;
    set_featureselection=para.set_featureselection;
end

resultAUC=[];
intFolds=para.intFolds;
intIter=para.intIter;
for i_c=1:length(set_classifier)
    for i_f=1:length(set_featureselection)
        para.feature_score_method='addone';
        para.classifier=set_classifier{i_c};
        para.num_top_feature=para.num_top_feature;
        para.featureranking=set_featureselection{i_f};
        para.correlation_factor=.95;
        [resultImbalancedC45,feat_scores] = ...
            nFold_AnyClassifier_withFeatureselection_v4(data_all,labels,feature_list_all,para,1,intFolds,intIter);
        
        %% get top ranked features
        [~,result_feat_idx_ranked{i_c,i_f}]=sort(feat_scores,'descend');
        result_feat_ranked{i_c,i_f}=feature_list_all(result_feat_idx_ranked{i_c,i_f});
        result_feat_scores{i_c,i_f}=feat_scores;
        %% get a blance sens and spec to report
        if para.get_balance_sens_spec
            for i=1:length(resultImbalancedC45)
                TPR=resultImbalancedC45(i).TPR;
                FPR=resultImbalancedC45(i).FPR;
                spe=1-FPR;
                labels=resultImbalancedC45(i).labels;
                
                balanceAcc=(spe+TPR)/2;
                [~,maxIdx]=max(balanceAcc);
                resultImbalancedC45(i).sens=TPR(maxIdx);
                resultImbalancedC45(i).spec=1-FPR(maxIdx);
                resultImbalancedC45(i).tp=round(resultImbalancedC45(i).sens*sum(labels));
                resultImbalancedC45(i).tn=round(resultImbalancedC45(i).spec*sum(~labels));
                resultImbalancedC45(i).fp=sum(~labels)-resultImbalancedC45(i).tn;
                resultImbalancedC45(i).fn=sum(labels)-resultImbalancedC45(i).tp;                
                resultImbalancedC45(i).acc=(resultImbalancedC45(i).tp+resultImbalancedC45(i).tn)/length(labels);
              %% modeified other metrics if neccesary !!
            end
        end
        % record the result here
        resultACC(i_c,i_f).mean_acc=mean([resultImbalancedC45.acc]);
        resultACC(i_c,i_f).std_acc=std([resultImbalancedC45.acc]);
        resultACC(i_c,i_f).mean_sens=mean([resultImbalancedC45.sens]);
        resultACC(i_c,i_f).std_sens=std([resultImbalancedC45.sens]);
        resultACC(i_c,i_f).mean_spec=mean([resultImbalancedC45.spec]);
        resultACC(i_c,i_f).std_spec=std([resultImbalancedC45.spec]);
        resultACC(i_c,i_f).mean_kappa = mean([resultImbalancedC45.kappa]);
        resultACC(i_c,i_f).std_kappa = std([resultImbalancedC45.kappa]);
        
        [resultAUC(i_c,i_f).max,idxMaxAUC_Im]=max([resultImbalancedC45.AUC]);
        [resultAUC(i_c,i_f).min,idxMinAUC_Im]=min([resultImbalancedC45.AUC]);
        resultAUC(i_c,i_f).mean=mean([resultImbalancedC45.AUC]);
        resultAUC(i_c,i_f).std=std([resultImbalancedC45.AUC]);
        structMaxAUC(i_c, i_f) = resultImbalancedC45(idxMaxAUC_Im);
    end
end
% save('4classifier_AUC_ACC_CV.mat', 'resultAUC','resultACC');
