function [methodstring,stats] = LDA( training_set , testing_set, training_labels, testing_labels )
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

methodstring = 'LDA';
try
    obj = fitcdiscr(training_set,training_labels);
catch
    obj = fitcdiscr(training_set,training_labels, 'DiscrimType', 'quadratic');
end
[~, probs] = predict(obj, testing_set);
stats.prediction = single(probs(:, 2));
if exist('testing_labels','var') && numel(unique(testing_labels)) > 1
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
end
