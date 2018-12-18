function [stats ] = run_demo_nFoldxVal( data_set , data_labels, n , kernel)

if nargin < 3
    kernel = 'linear';
end

%%sample data
% data_set = [ 1 2 3 4 5; 1 2 3 4 5; 1 2 3 4 5; 5 4 3 2 1; 5 4 3 2 1; 5 4 3 2 1];
% data_labels = [ -1 -1 -1 1 1 1]';

% Runs a demo on a given data set.
% This is an example on what should be done to train and predict on a data set. The information
% is not stored externally but it prints important information on screen. It gives a quick idea
% on how the SVM is doing.
% The data set is divided internally in the training set and testing set. Options are provided to
% select the training set randomly or just select a specific number of postive and negative examples
% for training.
%
% Parameters
%   data_set    list of eamples arranged by rows.
%       - N observations x P features
%   data_labels list of labels corresponding to the examples.
%       - labels [-1 -1 -1  1  1  1]'

data_labels = data_labels(:);
if ~length(find(data_labels == -1)), data_labels = (data_labels-1)*2-1; end


train_name = 'demonstration';

% reset total statistics
Ttp = 0; Ttn = 0; Tfp = 0; Tfn = 0;

% 1. First, we acquire the training set, training labels, testing set and testing labels.
%    For this, we will divide our data set in two. We will find positive (a)
%    and negative (b) examples to create a balanced training set.

a = find( data_labels >  0 );
b = find( data_labels <= 0 );

% commit a portion of the dataset for training
% a_values = 1:(size(a,1)/3);  %commit a portion of positive samples for training
% b_values = 1:(size(b,1)/3);   %commit a portion of negative samples for training

% commit a random portion of the dataset for training
a_shuffle = randperm(length(a));    %randomize index
b_shuffle = randperm(length(b));    %randomize index


% or don't shuffle
a_shuffle = 1:length(a);    %same index
b_shuffle = 1:length(b);    %same index

% define n sets
% a_cuts = round((1:n-1)/n*size(a,1));
a_cuts = [0 round((1:n-1)/n*length(a)) size(a,1)];
b_cuts = [0 round((1:n-1)/n*length(b)) size(b,1)];

for i=1:length(a_cuts)-1
    % commit 1 set from each class for testing
    a_values = a_shuffle(a_cuts(i)+1:a_cuts(i+1));
    b_values = b_shuffle(b_cuts(i)+1:b_cuts(i+1));
    
    % select training set
    %     training_set    = data_set( [ a(a_values) ; b(b_values) ] , : );
    %     training_labels = data_labels( [ a(a_values) ; b(b_values) ] , : );
    testing_set    = data_set( [ a(a_values) ; b(b_values) ] , : );
    testing_labels = data_labels( [ a(a_values) ; b(b_values) ]);
    % select random training set
    
    
    % select training by eliminating training samples from entire dataset
    training_set = data_set;
    training_labels = data_labels;
    
    training_set([ a(a_values) ; b(b_values) ] , : ) = [];
    training_labels([ a(a_values) ; b(b_values) ]) = [];
    
    % 2. Perform cross validation on the training set. This will return three of the best values
    %    (not necessarily the best three) so we choose the first value only.
    
    [ c , g ] = cv_svm( train_name ,  training_set , training_labels , kernel);
    c = c(1);
    g = g(1);
    
    % 3. Train the SVM with these parameters and the training set we selected. The training name
    %    will be the prefix to all of the generated files.
    train_svm( train_name , training_set , training_labels , c , g , kernel);
    
    % 4. Run the prediction on the test set using the generated training file.
    %    *It will return a vector with the distances from all the points to the
    %    hyper-plane.
    prediction = predict_svm(  train_name , testing_set );
    
    % 5. This prediction allows us to draw an ROC curve.
    %[ spec sens area ] = roc_svm( prediction , testing_labels );
    %plot( 1-spec , sens);
    
    % 6. And we can also calculate the accuracy.
    [ tp , tn , fp , fn ] = count_values( prediction , testing_labels ) ;
    Ttp = Ttp + tp;
    Ttn = Ttn + tn;
    Tfp = Tfp + fp;
    Tfn = Tfn + fn;
    %fprintf('Area:        %f\n' , area );
    % fprintf(' %5.3f ' , (tp+tn)/(tp+tn+fp+fn) );
    % fprintf(' %5.3f ', tp/(tp+fp));
    % fprintf(' %5.3f ' , tp/(tp+fn) );
    % fprintf(' %5.3f ' , tn/(tn+fp) );
    
    if ispc
        !del demonstration*.*
        !del decision_values.txt
    else
        system('rm demonstration*.*')
        % system('rm decision_values.txt')
    end
end

% output statistics
stats.tp = Ttp;
stats.tn = Ttn;
stats.fp = Tfp;
stats.fn = Tfn;
stats.acc = (Ttp+Ttn)/(Ttp+Ttn+Tfp+Tfn);
stats.ppv = Ttp/(Ttp+Tfp);
stats.sens = Ttp/(Ttp+Tfn);
stats.spec = Ttn/(Tfp+Ttn); 