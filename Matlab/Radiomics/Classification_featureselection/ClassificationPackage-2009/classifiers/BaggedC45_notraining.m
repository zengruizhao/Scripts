function [methodstring,stats] = BaggedC45_notraining(B, testing_set, testing_labels)

% format labels from (-1,1) to (0,1)
if any(testing_labels < 0)
    testing_labels(testing_labels==-1) = 0;
end

methodstring = 'BaggedC45';
% % [final, labels, error] = baggingdtrees(training_set, training_labels, testing_set, testing_labels);
% % 
% % 
% % tp=0; tn=0; fp=0; fn=0; prediction = zeros(size(final));
% % for i=1:size(labels,1)
% %     prediction(i) = round(mean(labels(i,:)));
% %     
% %     if testing_labels(i)==1
% %         if prediction(i) == testing_labels(i)
% %             tp = tp + 1;
% %         else
% %             fp = fp + 1;
% %         end
% %     elseif testing_labels(i)==2
% %         if prediction(i) == testing_labels(i)
% %             tn = tn + 1;
% %         else
% %             fn = fn + 1;
% %         end
% %     end
% % end
% 
% % Check for matlabpool
% try
% if matlabpool('size')==0, matlabpool open; end
% catch e
%     fprintf('Could not open matlabpool...skipping\n');
% end
% 
% % Options for TreeBagger
% options = statset('UseParallel','always','UseSubstreams','never');

% B = TreeBagger(50,training_set,training_labels,'FBoot',0.667,'oobpred','on','Method','classification','NVarToSample','all','NPrint',4,'Options',options);    % create bagged d-tree classifiers from training
[Yfit,Scores] = predict(B,testing_set);   % use to classify testing
probs = Scores(:,2); % Select probabilities -- check manual entry for 'predict', look at 'B' to make sure your reqd class is the second column
% if numel(unique(testing_labels))>1
    [FPR,TPR,T,AUC,OPTROCPT,~,~] = perfcurve(testing_labels,probs,'1');  % calculate AUC. 'perfcurve' can also calculate sens, spec etc. to plot the ROC curve.
    [TP FN] = perfcurve(testing_labels,probs,1,'xCrit','TP','yCrit','FN');
    [FP TN] = perfcurve(testing_labels,probs,1,'xCrit','FP','yCrit','TN');
    [~,ACC] = perfcurve(testing_labels,probs,1,'xCrit','TP','yCrit','accu');
    [~,PPV] = perfcurve(testing_labels,probs,1,'xCrit','TP','yCrit','PPV');
    
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
    stats.prediction = probs;
    stats.threshold = T(optim_idx);
    stats.decision = stats.prediction >= stats.threshold;
	stats.trained_classifier = B;
% end
    
    
