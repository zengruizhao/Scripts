function [methodstring,stats] = NBayes(training_set , testing_set, training_labels, testing_labels,varargin)

unq_tra_lab = unique(training_labels);
if numel(unq_tra_lab) ~= 2
    error('Only 2 labels allowed');
else
    idx1 = ismember(training_labels,unq_tra_lab(1));
    idx2 = ismember(training_labels,unq_tra_lab(2));
    training_labels(idx1) = 0;
    training_labels(idx2) = 1;
    idx1 = ismember(testing_labels,unq_tra_lab(1));
    idx2 = ismember(testing_labels,unq_tra_lab(2));
    testing_labels(idx1) = 0;
    testing_labels(idx2) = 1;
end

methodstring = 'NBayes';

% parse optional inputs
num_opt_args = length(varargin);
if num_opt_args >= 1 || ~isempty(varargin{1}{1})
    distrib = varargin{1}{1};
else
    distrib = 'normal'; % default to normal distribution
end
if num_opt_args >= 2 && ~isempty(varargin{2}{1})
    prior = varargin{2}{1};
else
    prior = 'empirical'; % default to empirical distribution
end

% expand 'distrib' if singular
if ischar(distrib)
    temp = cell(size(training_set,2),1);
    temp(:) = {distrib};
    distrib = temp;
    clear temp;
end

% fit training data using specified distribution and prior info
O1 = fitcnb(training_set,training_labels,'DistributionNames',distrib,'Prior',prior);

% predict testing data using trained classifier
% C1 = O1.predict(testing_set);

% evaluation

% this stuff gets saved regardless of the number of classes
% stats.posterior = posterior(O1, testing_set);
[C1, stats.posterior,~] = predict(O1, testing_set);
stats.trained_classifier = O1;
stats.distrib = distrib;
stats.prior = prior;
stats.prediction = stats.posterior(:,2);
unq_testing_labels = unique(testing_labels);
if numel(unq_testing_labels) == 2 % things we will do only if there are 2 classes
    stats.prediction = stats.posterior(:,2);
    [FPR,TPR,T,AUC,OPTROCPT,~,~] = perfcurve(testing_labels,stats.prediction,1);  % calculate AUC. 'perfcurve' can also calculate sens, spec etc. to plot the ROC curve.
    [TP, FN] = perfcurve(testing_labels,stats.prediction,1,'xCrit','TP','yCrit','FN');
    [FP, TN] = perfcurve(testing_labels,stats.prediction,1,'xCrit','FP','yCrit','TN');
    ACC = perfcurve(testing_labels,stats.prediction,1,'xCrit','accu');
    PPV = perfcurve(testing_labels,stats.prediction,1,'xCrit','PPV');
    
    optim_idx = find(FPR == OPTROCPT(1) & TPR == OPTROCPT(2));
    stats.tp = TP(optim_idx);
    stats.fn = FN(optim_idx);
    stats.fp = FP(optim_idx);
    stats.tn = TN(optim_idx);
    stats.auc = AUC;
    stats.spec = 1-FPR(optim_idx);
    stats.sens = TPR(optim_idx);
    stats.acc = ACC(optim_idx);
    stats.ppv = PPV(optim_idx);
    stats.threshold = T(optim_idx);
    stats.decision = stats.prediction >= stats.threshold;
	    
elseif numel(unq_testing_labels) > 2 % can't do ROC if there's more than 2 classes, so we'll do simple accuracy
    stats.decision = C1; % setting decision directly from bayesian classifier
    
    % calculating accuracy
    correctly_classified = 0;
    for i=1:length(unq_testing_labels)
        idx = testing_labels == unq_testing_labels(i);
        correctly_classified = correctly_classified + nnz(stats.decision(idx) == testing_labels(idx));
    end
    stats.acc = correctly_classified/length(testing_labels);
    
end