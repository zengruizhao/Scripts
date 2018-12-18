function [methodstring,stats] = ecoc( training_set , testing_set, training_labels, testing_labels)

methodstring = 'ecoc';
t=templateSVM('KernelFunction', 'linear');
ens = fitcecoc(training_set, training_labels, 'FitPosterior', 1, 'Learners', t); 
[~, ~, ~, probs] = predict(ens, testing_set);

stats.prediction = single(probs(:, 2));
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