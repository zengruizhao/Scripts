function [FPR TPR T AUC] = svm_roc(testing_labels,prob_estimates)
% USAGE: [FPR TPR T AUC] = svm_roc(testing_labels,prob_estimates)
% 
% Calculates ROC coordinates [FPR, TPR] by thresholding probability estimates (prob_estimates) 
% at values specified in T.  ROC curve can be visualized with plot(FPR,TPR).
% 
% Inputs:
%   testing_labels [Mx1 double]: Labels for M observations used for testing in SVM classifier. 
%   prob_estimates [Mx2 double]: Probability estimates returned from SVM classifier.  This is 
%                                identical to 'outputs.prob_estimates' returned by 'svm_main'.
% 
% Outputs:
%   FPR [Ax1 double]: False positive rates calculated from A different thresholds of prob_estimates.
%   TPR [Ax1 double]: True positive rates calculated from A different thresholds of prob_estimates.
%   T   [Ax1 double]: The A thresholds used to generate FPR and TPR.
%   AUC [double]: Area under the ROC curve.

[FPR, TPR, T, AUC] = perfcurve(testing_labels,prob_estimates(:,1),1);
