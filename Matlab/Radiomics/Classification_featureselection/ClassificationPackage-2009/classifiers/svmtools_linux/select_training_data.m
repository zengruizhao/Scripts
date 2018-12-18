function [training_set, training_labels, testing_set, testing_labels] = select_training_data(data_set, data_labels)

% 1. First, we acquire the training set, training labels, testing set and testing labels.      
 %    For this, we will divide our data set in two. We will find positive (a)
        %    and negative (b) examples to create a balanced training set.
fprintf('Selecting Training Data... ')

a = find( data_labels >  0 );
b = find( data_labels <= 0 );

% commit a portion of the dataset for training
a_values = 1:(size(a,1)/3);  %commit a portion of positive samples for training
b_values = 1:(size(b,1)/3);   %commit a portion of negative samples for training

% commit a random portion of the dataset for training
%  a_shuffle = randperm(size(a,1));    %randomize index
%  b_shuffle = randperm(size(b,1));    %randomize index
%  a_values = a_shuffle(1:size(a,1)/2); %commit portion of positive samples for training
%  b_values = b_shuffle(1:size(b,1)/2);  %commit portion of positive samples for training

% select training set
training_set    = data_set( [ a(a_values) ; b(b_values) ] , : );
training_labels = data_labels( [ a(a_values) ; b(b_values) ] , : );

% select testing and then eliminate the training examples
testing_set = data_set;
testing_labels = data_labels;

testing_set([ a(a_values) ; b(b_values) ] , : ) = [];
testing_labels([ a(a_values) ; b(b_values) ] , : ) = [];
fprintf('Done. \n')