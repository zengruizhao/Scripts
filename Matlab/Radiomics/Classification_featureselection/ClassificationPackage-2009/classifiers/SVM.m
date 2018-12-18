function [methodstring,outputs] = SVM(training_set, testing_set, training_labels, testing_labels, varargin)
%function [methodstring,outputs.tp,outputs.tn,outputs.fp,outputs.fn,outputs.prob_estimates] = SVM( training_set , testing_set, training_labels, testing_labels,params)

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
training_labels = training_labels(:);
testing_labels = testing_labels(:);

% if size(data_labels,1) == 1, data_labels = data_labels'; end
if nargin > 4 && isstruct(varargin{1})
    params = varargin{1};
    params.kernel = lower(params.kernel);
    switch params.kernel
        case {'linear','rbf','poly','sigmoid'}
        otherwise 
            fprintf('Invalid kernel selected.  Using linear kernel by default.\n');
            params.kernel = 'linear';
    end
else
    fprintf('No kernel selected.  Using linear kernel by default.\n');
    params.kernel = 'linear';
end

methodstring = ['SVM classifier (' params.kernel ' kernel) '];

% % 2. Perform cross validation on the training set. This will return three of the best values
% %    (not necessarily the best three) so we choose the first value only.
% if isunix
%      [ c , g ] = cv_svm( train_name ,  training_set , training_labels , kernel);
% elseif ispc
%    [ c , g ] = cv_svm_win( train_name ,  training_set , training_labels );
% end
% c = c(1);
% g = g(1);
% 
% % 3. Train the SVM with these parameters and the training set we selected. The training name
% %    will be the prefix to all of the generated files.
% if isunix
%     train_svm( train_name , training_set , training_labels , c , g ,kernel);
% elseif ispc
%     train_svm_win( train_name , training_set , training_labels , c , g ,kernel);
% end
% 
% % 4. Run the prediction on the test set using the generated training file.
% %    *It will return a vector with the distances from all the points to the
% %    hyper-plane.
% if isunix
%     prediction = predict_svm(  train_name , testing_set );
% elseif ispc
%     prediction = predict_svm_win(  train_name , testing_set );
% end
% 
% % 5. This prediction allows us to draw an ROC curve.
% %[ spec sens area ] = roc_svm( prediction , testing_labels );
% %plot( 1-spec , sens);
% 
% % 6. And we can also calculate the accuracy.
% [ tp , tn , fp , fn ] = count_values( prediction , testing_labels ) ;
% 
% %fprintf('Area:        %f\n' , area );
% % fprintf(' %5.3f ' , (tp+tn)/(tp+tn+fp+fn) );
% % fprintf(' %5.3f ', tp/(tp+fp));
% % fprintf(' %5.3f ' , tp/(tp+fn) );
% % fprintf(' %5.3f ' , tn/(tn+fp) );
% 
% % acc = (tp+tn)/(tp+tn+fp+fn);  %compute accuracy
% %ppv = tp/(tp+fp);
% %sens = tp/(tp+fn);
% %spec = tn/(fp+tn);
% 
% 
% if ispc
% !del demonstration*.*
% !del decision_values.txt
% else
%     system('rm demonstration*.*');
%     % system('rm decision_values.txt')
% end

outputs = svm_main(training_set, training_labels, testing_set, testing_labels,params);
