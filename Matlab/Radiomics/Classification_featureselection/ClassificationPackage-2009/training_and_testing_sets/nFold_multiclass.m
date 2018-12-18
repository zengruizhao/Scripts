function [training testing] = nFold(data_set, data_labels, shuffle, n)

data_labels = data_labels(:);

% 1. First, we acquire the training set, training labels, testing set and testing labels.
%    For this, we will divide our data set in two. We will find positive (a)
%    and negative (b) examples to create a balanced training set.

unique_labels = unique(data_labels);

for i = 1:length(unique_labels)
    idx{i} = find(data_labels == unique_labels(i));
end

% commit a random portion of the dataset for training
for i=1:length(unique_labels)
    if shuffle
        idx_shuffle{i} = randperm(length(idx{i})); % randomize index
    else % or don't shuffle
        idx_shuffle{i} = 1:length(idx{i}); % fixed index
    end
end
    
% define n sets
for i=1:length(unique_labels)
    cuts{i} = [0 round((1:n-1)/n*length(idx{i})) size(idx{i},1)];
end

testing=cell(1,n);
training=cell(1,n);
for i=1:n
    values = []; notvalues = [];
    for j=1:length(unique_labels)
        ind = idx_shuffle{j}(cuts{j}(i)+1:cuts{j}(i+1));
        values = [values; idx{j}(ind)];
        temp = idx{j};
        temp(ind) = [];
        notvalues = [notvalues; temp];
    end
%     a_values = a(a_ind);
%     b_values = b(b_ind);
%     values = [a_values ; b_values];
    
%     a_notvalues = a;
%     b_notvalues = b;
%     a_notvalues(a_ind) = [];
%     b_notvalues(b_ind) = [];
%     notvalues = [a_notvalues ; b_notvalues];
    
    % set testing set and labels
%     testing_set{i} = data_set(values,:);
    testing{i} = values;
%     testing_labels{i} = data_labels(values);
    
    % set training set as all samples not included in testing set
%     training_set{i} = data_set;
    training{i} = notvalues;
%     training_labels{i} = data_labels;
  
%     temp = training_set{i};
%     temp(values,:) = [];
%     training_set{i} = temp;
%     
%     temp = training_labels{i};
%     temp(values) = [];
%     training_labels{i} = temp;
end