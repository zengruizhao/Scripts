function stats = LnFold_BaggedC45_MES(data_set,data_labels,shuffle,n,nIter,Subsets)
% use Multi-expert system idea to reduce the effect of the imbalanced data set problem
% Nov. 13 2015 by Cheng LU
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
if nargin < 6
    Subsets = {};
end
if nargin < 5
    nIter = 1;
end
if nargin < 4
    n = 3; % 3-fold cross-validation
end
if nargin < 3
    shuffle = 1; % randomized
end

% if any(~xor(data_labels == 1, data_labels == -1)), error('Labels must be 1 and -1'); end

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
        %% !!!!!!!!!!! introduce MES here
        
        idxPositive = find( training_labels >  0 );
        idxNegative = find( training_labels <= 0 );
        % assume the negative is the majority class
        
        numME=floor(length(idxNegative)/length(idxPositive));
        % train numME classifiers
        
        for iME=1:numME
            curtraining_labels=[training_labels(idxPositive);training_labels(idxNegative((iME-1)*length(idxPositive)+1:(iME)*length(idxPositive)))];
            
            curtraining_set=[training_set(idxPositive,:);training_set(idxNegative((iME-1)*length(idxPositive)+1:(iME)*length(idxPositive)),:)];
            
            [temp_statsME(iME),methodstring] = Classify( 'BaggedC45', curtraining_set , testing_set, curtraining_labels(:), testing_labels(:));
            
        end
        statsME.prediction=mean([temp_statsME.prediction],2);
        statsME.decision=mean([temp_statsME.prediction],2)>0.5;
        
        if numel(unique(testing_labels)) > 1
            [FPR,TPR,T,AUC,OPTROCPT,~,~] = perfcurve(testing_labels,statsME.prediction,1);  % calculate AUC. 'perfcurve' can also calculate sens, spec etc. to plot the ROC curve.
            optim_idx = FPR == OPTROCPT(1) & TPR == OPTROCPT(2);
            statsME.auc = AUC;
            statsME.fpr = FPR;
            statsME.tpr = TPR;
            statsME.threshold = T(optim_idx);
        else
            statsME.auc = [];
            statsME.fpr = [];
            statsME.tpr = [];
            statsME.threshold = [];
        end

        trues = statsME.decision(:)' == testing_labels(:)'; % true results are those where decision matches labels
        positives = statsME.decision(:)' == 1; % positive results are those where decision is "1", i.e. the target class
        statsME.tp = nnz(trues & positives);
        statsME.tn = nnz(trues & ~positives);
        statsME.fp = nnz(~trues & positives);
        statsME.fn = nnz(~trues & ~positives);
        statsME.acc = (statsME.tp + statsME.tn) / (statsME.tp + statsME.fp + statsME.fn + statsME.tn);
        statsME.ppv = statsME.tp / (statsME.tp + statsME.fp);
        statsME.sens = statsME.tp / (statsME.tp + statsME.fn);
        statsME.spec = statsME.tn / (statsME.tn + statsME.fp);
        statsME.bacc = (statsME.sens + statsME.spec)/2;
        statsME.f1 = 2*statsME.ppv*statsME.sens/(statsME.ppv + statsME.sens);
        
        % average the soft prediction values for the MES
        prediction(tes{i})=statsME.prediction;
        decision(tes{i}) = statsME.decision;
        Ttp = Ttp + statsME.tp;
        Ttn = Ttn + statsME.tn;
        Tfp = Tfp + statsME.fp;
        Tfn = Tfn + statsME.fn;  
        clear temp_statsME
        clear statsME
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