function [stats, feature_scores]= nFold_BaggedC45_withFeatureselection(data_set,data_labels,feature_list,shuffle,n,nIter,Subsets)
% Using n-fold subset selection and C4.5 decision tree classifier
% Input:
%   data_set: data
%   data_labels: labels
%   shuffle: 1 for random, 0 for non-random partition (Default: 1)
%   n: Number of folds to your cross-validation (Default: 3)
%   nIter: Number of cross-validation iterations (Default: 1)
%   Subsets: pass your own training and testing subsets & labels (Default:
%   computer will generate using 'nFold')
%
% Output:
%   stats: struct containing TP, FP, TN, FN, etc.

% data_set = eigVec_graphfeats;
% data_labels = labels(:);
% classifier = 'SVM';
% shuffle = 1;
% n = 3;
if nargin < 7
    Subsets = {};
end
if nargin < 6
    nIter = 1;
end
if nargin < 5
    n = 4; % 3-fold cross-validation
end
if nargin < 4
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
        K=5;
        %% using mrmr
        % map the data in [0,100] and all integers
        %         dataw_discrete=makeDataDiscrete(training_set);
        %         setAll=1:size(training_set,2);
        %         [idx_TTest] = mrmr_mid_d(dataw_discrete(:,setAll), training_labels, K);
        %         setTopF_mRMR=feature_list(idx_TTest)';
        
        %% using ttest
        [TTidx,confidence] = prunefeatures_new(training_set, training_labels, 'ttestp');
        %         find(confidence<0.05);
        %         [TTidx,confidence] = prunefeatures_new(training_set, training_labels, 'wilcoxon');
        %         [Wilcoxonidx,confidence] = prunefeatures_new(dataw, labels, 'wilcoxon');
        
        idx_TTest=TTidx(confidence<0.05);
        if isempty(idx_TTest)
            idx_TTest=TTidx(1:5);
        end
        %%% lock down top features with low correlation
        set_candiF=Lpick_top_n_features_with_pvalue_correlation(training_set,idx_TTest,9);
        set_fff=feature_list(set_candiF)';
        idx_TTest=set_candiF;
        %% using forward sequential
%         [TTidx,confidence] = prunefeatures_new(training_set, training_labels, 'ttestp');
% %         set_potentialF_idx= confidence<0.05;
%         set_potentialF_idx=1:100;%find(confidence<0.02);
%         %          set_potentialF_idx=find(confidence<0.5);
%         % lulu
%         fs1=TTidx(set_potentialF_idx);
%         
%         threefoldCVP = cvpartition(training_labels,'kfold',3);
%         
%         % fun = @(XT,yT,Xt,yt)...
%         %       (sum(~strcmp(yt,classify(Xt,XT,yT,'quadratic'))));
%         %         try
%         %             fun = @(XT,yT,Xt,yt)...
%         %                 (sum(~strcmp(yt,classify(Xt,XT,yT,'linear'))));
%         %
%         %             fsLocal = sequentialfs(fun,training_set(:,fs1),training_labels','cv',threefoldCVP);
%         %         catch
%         %             fun = @(XT,yT,Xt,yt)...
%         %                 (sum(~strcmp(yt,classify(Xt,XT,yT,'diaglinear'))));
%         %             fsLocal = sequentialfs(fun,training_set(:,fs1),training_labels','cv',threefoldCVP);
%         %         end
% %         
% %         fun = @(XT,yT,Xt,yt)...
% %             (sum(~strcmp(yt,classify(Xt,XT,yT,'diaglinear'))));
%         fun = @(training_set,training_labels,testing_set,testing_labels)...
%             (sum(~strcmp(testing_labels,classf_RandomForest(training_set,testing_set,training_labels,testing_labels))));
%                
%         fsLocal = sequentialfs(fun,training_set(:,fs1),training_labels','cv',threefoldCVP);
%         
%         idx_TTest=fs1(fsLocal);
%         feature_list(idx_TTest)
        
        %% test on the testing set
        %         a=setTopF_TTest{1};b=setTopF_TTest{2};
        %         strcmp
        %         interr=intersect(a,b);
        % add one value on the piceked features
        feature_scores(idx_TTest)=feature_scores(idx_TTest)+1;
        
%                 [temp_stats,methodstring] = Classify( 'SVM', training_set(:,idx_TTest) , testing_set(:,idx_TTest), training_labels(:), testing_labels(:));
        
        [temp_stats,methodstring] = Classify( 'BaggedC45', training_set(:,idx_TTest) , testing_set(:,idx_TTest), training_labels(:), testing_labels(:));
%         [temp_stats,methodstring] = Classify( 'QDA', training_set(:,idx_TTest) , testing_set(:,idx_TTest), training_labels(:), testing_labels(:));
        
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