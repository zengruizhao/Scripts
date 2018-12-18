%% generate predictive labels
% how to set the para?
% para.featureranking='mrmr';
% para.num_top_feature=15;
% para.classifier_name='LDA';
% para.T_on_predicted_pro=0.5;
% para.feature_selection_mode='one-shot'; or 'cross-validation'

function [probability_pred,probability_prob, result, probability_pred_training]=Lgenerate_Predicted_Label_of_test_data_no_FS(training_set,testing_set,training_labels, testing_labels,para)
%% classification
if strcmp(para.classifier_name,'NBayes')
    distrib = para.distrib;
    prior = para.prior;
    temp_stats_testing = Classify(para.classifier_name, training_set , testing_set, training_labels(:), testing_labels(:), distrib, prior);
    temp_stats_training = Classify(para.classifier_name, training_set , training_set, training_labels(:), training_labels(:), distrib, prior);
elseif strcmp(para.classifier_name,'LDA')
    temp_stats_testing = Classify(para.classifier_name, training_set , testing_set, training_labels(:), testing_labels(:));
    temp_stats_training = Classify(para.classifier_name, training_set , training_set, training_labels(:), training_labels(:));
else
    temp_stats_testing = Classify(para.classifier_name, training_set , testing_set, training_labels(:), testing_labels(:));
    temp_stats_training = Classify(para.classifier_name, training_set , training_set, training_labels(:), training_labels(:));
end
probability_pred=temp_stats_testing.prediction;
probability_prob = temp_stats_testing.probs;
probability_pred_training = temp_stats_training.prediction;

% if size(labels_pred,1)~=size(test,1)
%     labels_pred=labels_pred';
% end
[FPR_train,TPR_train,T,AUC_train,OPTROCPT,~,~] = perfcurve(training_labels, probability_pred_training, 1);
[FPR_test,TPR_test,T,AUC_test,OPTROCPT,~,~] = perfcurve(testing_labels, probability_pred, 1);
result.testing.sen = temp_stats_testing.tp/(temp_stats_testing.tp + temp_stats_testing.fn);
result.testing.spe = temp_stats_testing.tn/(temp_stats_testing.tn + temp_stats_testing.fp);
result.testing.auc = AUC_test;
result.testing.FPR = FPR_test;
result.testing.TPR = TPR_test;
%
result.training.sen = temp_stats_training.tp/(temp_stats_training.tp + temp_stats_training.fn);
result.training.spe = temp_stats_training.tn/(temp_stats_training.tn + temp_stats_training.fp);
result.training.auc = AUC_train;
result.training.FPR = FPR_train;
result.training.TPR = TPR_train;
end
