function [scaled_data indice]= svm_scale_no_clear(data)
% USAGE: scaled_data = svm_scale(data)
% 
% This function scales a dataset so that each column (i.e. feature) is
% normalized to zero mean and unit standard deviation.  It should be used
% to scale a dataset before performing any SVM-related tasks.
% 
% Inputs:
%   data [NxP double]: Input dataset containing N observations and P features.
% 
% Outputs:
%   scaled_data [NxP double]: Scaled dataset containing N observations and P features.

data = bsxfun(@minus,data,mean(data)); % zero mean
scaled_data = bsxfun(@rdivide,data,std(data)); % unit std dev
