function [methodstring,stats] = svmmine( training_set , testing_set, training_labels, testing_labels)

methodstring = 'svm';

ens = fitcsvm(training_set, training_labels, 'KernelFunction', 'rbf');
ens = compact(ens);
[~, probs] = predict(ens, testing_set);
[ens, parameters] = fitPosterior(ens, training_set, training_labels); 
[~,validation_probs] = predict(ens, testing_set);

stats.prediction = single(validation_probs(:, 2));
stats.probs = single(probs(:, 1));
if exist('testing_labels','var') && numel(unique(testing_labels)) > 1
    [FPR,TPR,T,AUC,OPTROCPT,~,~] = perfcurve(testing_labels,stats.prediction,1);  % calculate AUC. 'perfcurve' can also calculate sens, spec etc. to plot the ROC curve.
    [TP, FN] = perfcurve(testing_labels,stats.prediction,1,'xCrit','TP','yCrit','FN');
    [FP, TN] = perfcurve(testing_labels,stats.prediction,1,'xCrit','FP','yCrit','TN');    
    optim_idx = find(FPR == OPTROCPT(1) & TPR == OPTROCPT(2));
    stats.tp = TP(optim_idx);
    stats.fn = FN(optim_idx);
    stats.fp = FP(optim_idx);
    stats.tn = TN(optim_idx);
end