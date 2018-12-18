function  results = run_decision_tree( name , train_data ,train_labels , test_data , test_labels )
% Runs the decision trees and outputs the number
% true positives, true negatives, false positives
% and false negatives.
%
% Parameters
%   name         : string with the name of the files to be generated.
%   train_data   : data to be used for training
%   train_labels : labels for the training data
%   test_data    : testing data
%   test_labels  : labels for the testing data
%
% Output
%    very long string with the outuput of the c4.5.exe program
%    it shows the decision tree and other information. The
%    table with the statistics it at the end.
%
% Example
%
%   results = run_decision_tree( 'prostate' , prostate_train , ...
%       prostate_train_labels , prostate_test, prostate_test_labels );
%

%disp('Writing data sets into output files');
write_data_set( name , train_data , train_labels , test_data , test_labels );
%disp('Finished writing');

% [ status results ] = unix( [ './c4.5 -f ' name ' -u ' ] );
[ status results ] = unix( [ './c4.5 -f ' name ' -u >./decision_trees_result.txt ' ]);
[ status results ] = unix( 'grep ITEM decision_trees_result.txt > ./decision_trees_result2.txt');
!sed -e 's/ITEM/ /g' decision_trees_result2.txt > ./decision_trees_result3.txt

incorrect_predictions = load( 'decision_trees_result3.txt' );
test_labels( incorrect_predictions+1 ) = test_labels(incorrect_predictions+1 ) * -1;

