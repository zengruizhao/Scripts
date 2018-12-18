function [tp, tn, fp, fn, test_labels, testing_labels] = decision_tree_multiclass(data_set, data_labels)
%formatting
name = 'Prostatecancertest'; % Don't use spaces in your names
%data_labels = (((data_labels-1)*2)-1)'; %transforms data labels 1,2 to -1,1

% 1. First, we acquire the training set, training labels, testing set and testing labels.
%    For this, we will divide our data set in two. We will find positive and negative
%    examples to create a balanced training set.

a = find( data_labels == 1 );
b = find( data_labels == 2 );
c = find( data_labels == 3 );
d = find( data_labels == 4 );
e = find( data_labels == 5 );


% a_values = ceil(rand(1,ceil(size(a,1)/4))*size(a,1)) ;
% b_values = ceil(rand(1,ceil(size(b,1)/4))*size(b,1))  ;
a_values = 1:(size(a,1)/3) ;
b_values = 1:(size(b,1)/3) ;
c_values = 1:(size(c,1)/3) ;
d_values = 1:(size(d,1)/3) ;
e_values = 1:(size(e,1)/3) ;

% select training set
training_set    = data_set( [ a(a_values) ; b(b_values); c(c_values); d(d_values); e(e_values) ] , : );
training_labels = data_labels( [ a(a_values) ; b(b_values); c(c_values); d(d_values); e(e_values) ] , : );

% select testing and then eliminate the training examples
testing_set = data_set;
testing_labels = data_labels;

testing_set([  a(a_values) ; b(b_values); c(c_values); d(d_values); e(e_values) ] , : ) = [];
testing_labels([  a(a_values) ; b(b_values); c(c_values); d(d_values); e(e_values) ] , : ) = [];

test_labels = run_decision_tree2( name , training_set , training_labels , testing_set , testing_labels );
[tp, tn, fp, fn] = count_values_multiclass(test_labels, testing_labels);
