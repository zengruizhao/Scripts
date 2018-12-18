function [tp,tn,fp,fn,prediction ] = run_demo( data_set , labels)

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

train_name = 'demonstration';

data_labels = labels;

% select testing and then eliminate the training examples
at = find(data_labels == 1);
bt = find(data_labels == 2);
ct = find(data_labels == 3);
dt = find(data_labels == 4);
et = find(data_labels == 5);
at_values = 1:(size(at,1)/3);  %commit a portion of positive samples for training
bt_values = 1:(size(bt,1)/3);   %commit a portion of negative samples for training
ct_values = 1:(size(ct,1)/3);  %commit a portion of positive samples for training
dt_values = 1:(size(dt,1)/3);   %commit a portion of negative samples for training
et_values = 1:(size(et,1)/3);  %commit a portion of positive samples for training

testing_set = data_set;
testing_labels = data_labels;
testing_set([ at(at_values) ; bt(bt_values); ct(ct_values); dt(dt_values); et(et_values) ] , : ) = [];
testing_labels([ at(at_values) ; bt(bt_values); ct(ct_values); dt(dt_values); et(et_values) ] , : ) = [];
prediction_table = ones(1,length(testing_labels));

m = max(labels);    
for i = 1:m
    data_labels = labels;
    a = find(data_labels == i);
    data_labels = ones(1,length(labels))';
    data_labels(a) = -1;

    % commit a portion of the dataset for training
    %         a_values = 1:(size(a,1)/3);  %commit a portion of positive samples for training
    %         b_values = 1:(size(b,1)/3);   %commit a portion of negative samples for training

    % commit a random portion of the dataset for training
    % a_shuffle = randperm(size(as,1));    %randomize index
    % b_shuffle = randperm(size(b,1));    %randomize index
    % a_values = a_shuffle(1:size(a,1)/3)  %commit portion of positive samples for training
    % b_values = b_shuffle(1:size(b,1)/3)  %commit portion of positive samples for training

    % select training set
    training_set    = data_set([ at(at_values) ; bt(bt_values); ct(ct_values); dt(dt_values); et(et_values) ] , : );
    training_labels = data_labels([ at(at_values) ; bt(bt_values); ct(ct_values); dt(dt_values); et(et_values) ] , : );

    % select random training set


    % 2. Perform cross validation on the training set. This will return three of the best values
    %    (not necessarily the best three) so we choose the first value only.

    [ c , g ] = cv_svm( train_name ,  training_set , training_labels );
    c = c(1);
    g = g(1);

    % 3. Train the SVM with these parameters and the training set we selected. The training name
    %    will be the prefix to all of the generated files.
    train_svm( train_name , training_set , training_labels , c , g );

    % 4. Run the prediction on the test set using the generated training file.
    %    *It will return a vector with the distances from all the points to the
    %    hyper-plane.
    prediction = predict_svm(  train_name , testing_set );

    prediction_table(i,:) = randn(1)+10;
    neg = find(sign(prediction) == -1);
    prediction_table(i,neg) = i;
end
    prediction = mode(prediction_table)
% 5. This prediction allows us to draw an ROC curve.
%[ spec sens area ] = roc_svm( prediction , testing_labels );
%plot( 1-spec , sens);

% 6. And we can also calculate the accuracy.
[ tp , tn , fp , fn ] = count_values_multiclass( prediction , testing_labels ) 

%fprintf('Area:        %f\n' , area );
% fprintf(' %5.3f ' , (tp+tn)/(tp+tn+fp+fn) );
% fprintf(' %5.3f ', tp/(tp+fp));
% fprintf(' %5.3f ' , tp/(tp+fn) );
% fprintf(' %5.3f ' , tn/(tn+fp) );

%acc = (tp+tn)/(tp+tn+fp+fn);  %compute accuracy
%ppv = tp/(tp+fp);
%sens = tp/(tp+fn);
%spec = tn/(fp+tn);


if ispc
    !del demonstration*.*
    !del decision_values.txt
else
    system('rm demonstration*.*')
    % system('rm decision_values.txt')
end
