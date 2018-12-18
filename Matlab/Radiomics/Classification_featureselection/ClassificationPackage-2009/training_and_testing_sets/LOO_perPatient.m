function [training testing] = LOO(data_set, data_labels, patient_labels)

data_labels = data_labels(:);
patient_labels = patient_labels(:);


% get unique patient labels and corresponding data labels
[unique_patient_labels, idx, ~] = unique(patient_labels,'first');
data_labels_perPatient = data_labels(idx);

[tra tes]=GenerateSubsets('LOO',[],data_labels_perPatient);

training = cellfun(@(x) ismember(patient_labels,x), tra, 'UniformOutput', false);
testing = cellfun(@(x) ismember(patient_labels,x), tes, 'UniformOutput', false);

% 
% n = length(data_labels);
% 
% 
% 
% testing=cell(1,n);
% training=cell(1,n);
% 
% for i=1:n
%     % set testing set and labels
%     testing{i} = i;
%     
%     % set training set as all samples not included in testing set
%     temp = 1:n;
%     temp(i)=[];
%     training{i}=temp;
% end