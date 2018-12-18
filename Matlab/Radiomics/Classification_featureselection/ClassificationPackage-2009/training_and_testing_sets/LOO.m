function [training testing] = LOO(data_set, data_labels)

data_labels = data_labels(:);

n = length(data_labels);

testing=cell(1,n);
training=cell(1,n);

for i=1:n
    % set testing set and labels
    testing{i} = i;
    
    % set training set as all samples not included in testing set
    temp = 1:n;
    temp(i)=[];
    training{i}=temp;
end