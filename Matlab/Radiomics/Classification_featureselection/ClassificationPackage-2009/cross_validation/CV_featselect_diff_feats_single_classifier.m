%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is 1) pick top feature from each feature set then 2)
% combine the feature to build clasifier and 3)test classifier under
% cross-validation scheme

% Input:
%   -feature_set_name   name of different feature set
%   -feature_set_data   a cell structure contains data from different
%                       feature set
%   -data_label     label for each sample
%   -para               see example below
%   -n                  folds
%   -nIter              iterations


% Output:
%   
% data_labels=labels_GT_trian;
% para.feature_score_method='addone';
% para.classifier='LDA';
% para.num_top_feature_in_each_feat_set=5;
% para.featureranking='wilcoxon';
% para.correlation_factor=.9;
% intFolds=5;
% intIter=10;
% 
% [stats, feature_scores]= CV_featselect_diff_feats_single_classifier(feature_set_name,feature_set_data, data_labels,para,shuffle,intFolds,intIter,Subsets);
%  

%   
% (c) Edited by Cheng Lu, 
% Biomedical Engineering,
% Case Western Reserve Univeristy, cleveland, OH
% If you have any problem feel free to contact me.
% Please address questions or comments to: hacylu@yahoo.com

% Terms of use: You are free to copy,
% distribute, display, and use this work, under the following
% conditions. (1) You must give the original authors credit. (2) You may
% not use or redistribute this work for commercial purposes. (3) You may
% not alter, transform, or build upon this work. (4) For any reuse or
% distribution, you must make clear to others the license terms of this
% work. (5) Any of these conditions can be waived if you get permission
% from the authors.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [stats, feature_scores_all_features]= CV_featselect_diff_feats_single_classifier(feature_set_name,feature_set_data, data_labels,para,shuffle,n,nIter,Subsets)
% need to ensure the label is in double format, otherwise we will have
% errors
data_labels=double(data_labels);

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

if size(feature_set_data{1},1)~=length(data_labels)
    error('the size of the feature data should be the same as the label data!!!');
end

%%% pre allocate the feature scores
feature_scores_all_features=[];
for i_feat=1:length(feature_set_name)
    data_set=feature_set_data{i_feat};
    feature_scores_all_features{i_feat}=zeros(size(data_set,2),1);
end

stats = struct; %cell(1,nIter);
for j=1:nIter
    fprintf('Iteration: %i\n',j);
    
    % reset total statistics
    Ttp = 0; Ttn = 0; Tfp = 0; Tfn = 0; 
    
    
    if isempty(Subsets)
        [tra tes]=GenerateSubsets('nFold',data_set,data_labels,shuffle,n);
        decision=zeros(size(data_labels)); prediction=zeros(size(data_labels));
    else
        tra{1} = Subsets{j}.training;
        tes{1} = Subsets{j}.testing;
%         decision=zeros(size(tes{1},1),1); prediction=zeros(size(tes{1}),1);
    end
    
    for i=1:n
        fprintf(['Fold # ' num2str(i) '\n']);
        training_labels = data_labels(tra{i});
        testing_labels = data_labels(tes{i});
        data_set_all_feats_train=[];
        data_set_all_feats_test=[];
%         idx_top_feats=[];
        
        for i_feat=1:length(feature_set_name)
            data_set=feature_set_data{i_feat};
            training_set = data_set(tra{i},:);
            testing_set = data_set(tes{i},:);
            
            feature_scores=feature_scores_all_features{i_feat};
            %% do feature selection on the fly from each feature set
            %% using mrmr
            if strcmp(para.featureranking,'mrmr')
                %         map the data in to binary values 0 1
                dataw_discrete=makeDataDiscrete_mrmr(training_set);
                %             dataw_discrete=training_set>t; check check check
                setAll=1:size(training_set,2);
                [idx_TTest] = mrmr_mid_d(dataw_discrete(:,setAll), training_labels, para.num_top_feature_in_each_feat_set);
            end
            %% using random forest
            if strcmp(para.featureranking,'rf')
                options = statset('UseParallel','never','UseSubstreams','never');
                B = TreeBagger(50,training_set,training_labels,'FBoot',0.667, 'oobpred','on','OOBVarImp', 'on', 'Method','classification','NVarToSample','all','NPrint',4,'Options',options);
                variableimportance = B.OOBPermutedVarDeltaError;
                [t,idx]=sort(variableimportance,'descend');
                idx_TTest=idx(1:para.num_top_feature_in_each_feat_set);
            end
            
            if strcmp(para.featureranking,'ttest') || strcmp(para.featureranking,'wilcoxon')
                %% using ttest
                if strcmp(para.featureranking,'ttest')
                    [TTidx,confidence] = prunefeatures_new(training_set, training_labels, 'ttestp');
                    idx_TTest=TTidx(confidence<0.05);
                    if isempty(idx_TTest)
                        idx_TTest=TTidx(1:min(para.num_top_feature_in_each_feat_set*2,size(data_set,2)));
                    end
                end
                
                if strcmp(para.featureranking,'wilcoxon')
                    [TTidx,confidence] = prunefeatures_new(training_set, training_labels, 'wilcoxon');
                    idx_TTest=TTidx(confidence<0.3);
                    if isempty(idx_TTest)
                        idx_TTest=TTidx(1:min(para.num_top_feature_in_each_feat_set*3,size(data_set,2)));
                    end
                end
                %%% lock down top features with low correlation
                set_candiF=Lpick_top_n_features_with_pvalue_correlation(training_set,idx_TTest,para.num_top_feature_in_each_feat_set,para.correlation_factor);
%                 set_fff=feature_list(set_candiF)'; % training_set(:,373)
                idx_TTest=set_candiF;
            end

            fprintf('on the fold, %d %s features are picked\n', length(idx_TTest),feature_set_name{i_feat});                        
            if  strcmp(para.feature_score_method,'addone')
                % add one value on the piceked features
                feature_scores(idx_TTest)=feature_scores(idx_TTest)+1;
            end
            
            if  strcmp(para.feature_score_method,'weighted')
                feature_scores(idx_TTest)=feature_scores(idx_TTest)+ linspace( para.num_top_feature_in_each_feat_set ,1, length(idx_TTest))';
            end
            

            feature_scores_all_features{i_feat}=feature_scores_all_features{i_feat}+feature_scores;

            %% gathering top features from different feature sets
            data_set_all_feats_train=cat(2,data_set_all_feats_train,training_set(:,idx_TTest));
%             idx_top_feats{i_feat}=idx_TTest;
            %% preparing testing data
            data_set_all_feats_test=cat(2,data_set_all_feats_test,testing_set(:,idx_TTest));
        end
        
        %% test on the testing set   
        try
            if strcmp(para.classifier,'BaggedC45')
                [temp_stats,methodstring] = Classify( 'BaggedC45', data_set_all_feats_train , data_set_all_feats_test, training_labels(:), testing_labels(:));
            end
            
            if strcmp(para.classifier,'QDA')|| strcmp(para.classifier,'qda')
                [temp_stats,methodstring] = Classify( 'QDA', data_set_all_feats_train , data_set_all_feats_test, training_labels(:), testing_labels(:));
            end
            
            if strcmp(para.classifier,'LDA') ||strcmp(para.classifier,'lda')
                [temp_stats,methodstring] = Classify( 'LDA', data_set_all_feats_train , data_set_all_feats_test, training_labels(:), testing_labels(:));
            end
            
            if strcmp(para.classifier,'SVM')||strcmp(para.classifier,'svm')
                if exist('para.params','var')
                    params.kernel=para.params.kernel;
                    params.c_range=para.params.c_range;
                    params.g_range=para.params.g_range;
                    params.cvfolds=para.params.cvfolds;
                    [temp_stats,methodstring] = Classify( 'SVM', data_set_all_feats_train , data_set_all_feats_test, training_labels(:), testing_labels(:),params);
                    
                else
                    [temp_stats,methodstring] = Classify( 'SVM', data_set_all_feats_train , data_set_all_feats_test, training_labels(:), testing_labels(:));
                end
                temp_stats.decision=temp_stats.predicted_labels;
                temp_stats.prediction=temp_stats.prob_estimates(:,1);
            end
        catch
            display('Error while using LDA or QDA, the training data is linear dependent, use Random Forest for this fold instead\n');
            [temp_stats,methodstring] = Classify( 'BaggedC45', data_set_all_feats_train , data_set_all_feats_test, training_labels(:), testing_labels(:));
        end
        Ttp = Ttp + temp_stats.tp;
        Ttn = Ttn + temp_stats.tn;
        Tfp = Tfp + temp_stats.fp;
        Tfn = Tfn + temp_stats.fn;
        if ~isempty(Subsets)
            stats=temp_stats;
            return;
        end
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
        
    % get a blance sens and spec to report
    if para.get_balance_sens_spec  
        spe=1-FPR;
        labels=stats(j).labels;
        balanceAcc=(spe+TPR)/2;
        [~,maxIdx]=max(balanceAcc);
        stats(j).sens=TPR(maxIdx);
        stats(j).spec=1-FPR(maxIdx);
        stats(j).tp=round(stats(j).sens*sum(labels));
        stats(j).tn=round(stats(j).spec*sum(~labels));
        stats(j).fp=sum(~labels)-stats(j).tn;
        stats(j).fn=sum(labels)-stats(j).tp;
        stats(j).acc=(stats(j).tp+stats(j).tn)/length(labels);
        %% modeified other metrics if neccesary !!

    end
end