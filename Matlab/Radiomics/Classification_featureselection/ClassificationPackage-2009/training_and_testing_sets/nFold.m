function [training, testing] = nFold(data_set, data_labels, shuffle, n)

data_labels = data_labels(:);

% 1. First, we acquire the training set, training labels, testing set and testing labels.
%    For this, we will divide our data set in two. We will find positive (a)
%    and negative (b) examples to create a balanced training set.

a = find( data_labels >  0 );
b = find( data_labels <= 0 );

if shuffle
    % commit a random portion of the dataset for training
    a_shuffle = randperm(length(a));    %randomize index
    b_shuffle = randperm(length(b));    %randomize index
else
    % or don't shuffle
    a_shuffle = 1:length(a);    %same index
    b_shuffle = 1:length(b);    %same index
end

% define n sets
a_cuts = [0 round((1:n-1)/n*length(a)) size(a,1)];
b_cuts = [0 round((1:n-1)/n*length(b)) size(b,1)];

testing=cell(1,n);
training=cell(1,n);
for i=1:n
    a_ind = a_shuffle(a_cuts(i)+1:a_cuts(i+1));
    b_ind = b_shuffle(b_cuts(i)+1:b_cuts(i+1));
    a_values = a(a_ind);
    b_values = b(b_ind);
    values = [a_values ; b_values];
    
    a_notvalues = a;
    b_notvalues = b;
    a_notvalues(a_ind) = [];
    b_notvalues(b_ind) = [];
    notvalues = [a_notvalues ; b_notvalues];
    
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
