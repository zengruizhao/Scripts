function [stats, feature_scores]= nFold_AnyClassifier_withFeatureselection_v2(data_set,data_labels,feature_list,para,shuffle,n,nIter,Subsets)
% Using n-fold subset selection and C4.5 decision tree classifier
% Input:
%   data_set: data
%   data_labels: labels
%   feature_list: the feature name list in cell
%   para: 
%    parameter like what classifier you use, the number of top feature
%    para.classifier='LDA';
%    para.num_top_feature=5;
%   shuffle: 1 for random, 0 for non-random partition (Default: 1)
%   n: Number of folds to your cross-validation (Default: 3)
%   nIter: Number of cross-validation iterations (Default: 1)
%   Subsets: pass your own training and testing subsets & labels (Default:
%   computer will generate using 'nFold')
%
% Output:
%   stats: struct containing TP, FP, TN, FN, etc.
% The function is written by Cheng Lu @2016
% example:
% para.classifier='LDA';
% para.num_top_feature=5;
% [resultImbalancedC45,feature_scores] = nFold_AnyClassifier_withFeatureselection(data_all_w,labels,feature_list_t,para,1,intFolds,intIter);


% data_set = eigVec_graphfeats;
% data_labels = labels(:);
% classifier = 'SVM';
% shuffle = 1;
% n = 3;
if nargin < 8
    Subsets = {};
end
if nargin < 7
    nIter = 1;
end
if nargin < 6
    n = 4; % 3-fold cross-validation
end
if nargin < 5
    shuffle = 1; % randomized
end

% if any(~xor(data_labels == 1, data_labels == -1)), error('Labels must be 1 and -1'); end
feature_scores=zeros(size(data_set,2),1);

stats = struct; %cell(1,nIter);
for j=1:nIter
    fprintf('Iteration: %i\n',j);
    
    % reset total statistics
    Ttp = 0; Ttn = 0; Tfp = 0; Tfn = 0; decision=zeros(size(data_labels)); prediction=zeros(size(data_labels));
    
    if isempty(Subsets)
        [tra tes]=GenerateSubsets('nFold',data_set,data_labels,shuffle,n);
    else
        tra = Subsets{j}.training;
        tes = Subsets{j}.testing;
    end
    
    for i=1:n
        fprintf(['Fold # ' num2str(i) '\n']);
        
        training_set = data_set(tra{i},:);
        testing_set = data_set(tes{i},:);
        training_labels = data_labels(tra{i});
        testing_labels = data_labels(tes{i});
        
        %%% do feature selection on the fly
        %% using mrmr
         if strcmp(para.featureranking,'mrmr') 
             %         map the data in [0,100] and all integers
             dataw_discrete=makeDataDiscrete(training_set);
             setAll=1:size(training_set,2);
             [idx_TTest] = mrmr_mid_d(dataw_discrete(:,setAll), training_labels, para.num_top_feature);
         end
         
         if strcmp(para.featureranking,'ttest') | strcmp(para.featureranking,'wilcoxon')
             %% using ttest
             if strcmp(para.featureranking,'ttest')
                 [TTidx,confidence] = prunefeatures_new(training_set, training_labels, 'ttestp');
                 idx_TTest=TTidx(confidence<0.05);
                 if isempty(idx_TTest)
                     idx_TTest=TTidx(1:para.num_top_feature*2);
                 end
             end
             
             if strcmp(para.featureranking,'wilcoxon')
                 [TTidx,confidence] = prunefeatures_new(training_set, training_labels, 'wilcoxon');
                 idx_TTest=TTidx(confidence<0.1);
                 if isempty(idx_TTest)
                     idx_TTest=TTidx(1:para.num_top_feature*2);
                 end
             end
             
             %%% lock down top features with low correlation
             set_candiF=Lpick_top_n_features_with_pvalue_correlation(training_set,idx_TTest,para.num_top_feature,para.correlation_factor);
             set_fff=feature_list(set_candiF)';
             idx_TTest=set_candiF;
         end  
        %% test on the testing set
        %         a=setTopF_TTest{1};b=setTopF_TTest{2};
        %         strcmp
        %         interr=intersect(a,b);
        % add one value on the piceked features
        feature_scores(idx_TTest)=feature_scores(idx_TTest)+1;
        fprintf('on the fold, %d features are picked\n', length(idx_TTest));
        try
            if strcmp(para.classifier,'BaggedC45')
                [temp_stats,methodstring] = Classify( 'BaggedC45', training_set(:,idx_TTest) , testing_set(:,idx_TTest), training_labels(:), testing_labels(:));
            end
            
            if strcmp(para.classifier,'QDA')
                [temp_stats,methodstring] = Classify( 'QDA', training_set(:,idx_TTest) , testing_set(:,idx_TTest), training_labels(:), testing_labels(:));
            end
            
            if strcmp(para.classifier,'LDA')
                [temp_stats,methodstring] = Classify( 'LDA', training_set(:,idx_TTest) , testing_set(:,idx_TTest), training_labels(:), testing_labels(:));
            end
            
            if strcmp(para.classifier,'SVM')
                if exist('para.params','var')
                    params.kernel=para.params.kernel;
                    params.c_range=para.params.c_range;
                    params.g_range=para.params.g_range;
                    params.cvfolds=para.params.cvfolds;
                    [temp_stats,methodstring] = Classify( 'SVM', training_set(:,idx_TTest) , testing_set(:,idx_TTest), training_labels(:), testing_labels(:),params);

                else
                    [temp_stats,methodstring] = Classify( 'SVM', training_set(:,idx_TTest) , testing_set(:,idx_TTest), training_labels(:), testing_labels(:));
                end
                temp_stats.decision=temp_stats.predicted_labels;
                temp_stats.prediction=temp_stats.prob_estimates(:,1);
            end            
        catch
            display('Error while using LDA or QDA, the training data is linear dependent, use Random Forest for this fold instead\n');
            [temp_stats,methodstring] = Classify( 'BaggedC45', training_set(:,idx_TTest) , testing_set(:,idx_TTest), training_labels(:), testing_labels(:));
        end
        Ttp = Ttp + temp_stats.tp;
        Ttn = Ttn + temp_stats.tn;
        Tfp = Tfp + temp_stats.fp;
        Tfn = Tfn + temp_stats.fn;

        decision(tes{i}) = temp_stats.decision;

        %       decision(tes{i}) = temp_stats.prediction >= temp_stats.threshold;
        prediction(tes{i}) = temp_stats.prediction;
    end
    decision(decision==0) = -1;
    
    % output statistics
    if numel(unique(data_labels))>1 %numel(unique(testing_labels))>1
        if n == 1
            [FPR,TPR,T,AUC,OPTROCPT,~,~] = perfcurve(data_labels(tes{i}),prediction(tes{i}),1);
        else
            [FPR,TPR,T,AUC,OPTROCPT,~,~] = perfcurve(data_labels,prediction,1);
        end
        stats(j).AUC = AUC;
        stats(j).TPR = TPR;
        stats(j).FPR = FPR;
    else
        stats(j).AUC = [];
        stats(j).TPR = [];
        stats(j).FPR = [];
    end
    
    stats(j).tp = Ttp;
    stats(j).tn = Ttn;
    stats(j).fp = Tfp;
    stats(j).fn = Tfn;
    stats(j).acc = (Ttp+Ttn)/(Ttp+Ttn+Tfp+Tfn);
    stats(j).ppv = Ttp/(Ttp+Tfp);
    stats(j).sens = Ttp/(Ttp+Tfn);
    stats(j).spec = Ttn/(Tfp+Ttn);
    stats(j).subsets.training = tra;
    stats(j).subsets.testing = tes;
    stats(j).labels = data_labels;
    stats(j).decision = decision;
    stats(j).prediction = prediction;
    Pre = ((Ttp+Tfp)*(Ttp+Tfn) + (Ttn+Tfn)*(Ttn+Tfp)) / (Ttp+Ttn+Tfp+Tfn)^2;
    stats(j).kappa = (stats(j).acc - Pre) / (1 - Pre);
end