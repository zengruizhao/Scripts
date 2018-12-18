function [scaled_data indice, mean_value, std_value]= svm_scale(data)
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
mean_value = mean(data);
data = bsxfun(@minus,data,mean(data)); % zero mean
std_value = std(data);
scaled_data = bsxfun(@rdivide,data,std(data)); % unit std dev
% scaled_data = bsxfun(@rdivide, data - min(data), max(data) - min(data));
% data = bsxfun(@minus,data,min(data));
% scaled_data = bsxfun(@rdivide, data, max(data) - min(data));
% scaled_data = data;
%% clear out nan
temp = scaled_data;
indice = [];j=1;
for i = 1:size(temp, 2)
    if ~isempty(find(unique(isnan(temp(:,i)))==1))
        fprintf('nan feature %d\n',i);
        indice(j) = i;
        scaled_data(:, i) = 0;
        j =j+1;
    end
end
% scaled_data(:,indice) = [];
