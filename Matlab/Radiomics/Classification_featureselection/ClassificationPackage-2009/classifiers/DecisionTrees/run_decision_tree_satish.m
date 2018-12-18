function  test_labels = run_decision_tree( name,train_data,train_labels,test_data,test_labels)%,f)
% Runs the decision trees and outputs the number
% true positives, true negatives, false positives
% and false negatives.
% CAN CURRENTLY ONLY BE RUN ON LINUX
%  WRITES TEMP FILES TO /HOME/SATISH/TEMP/$NAME$/
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
% name = [name,num2str(f)];

currdir = cd;
outdir = [currdir '/' name]; outname = [outdir,'/',name];
[foo,foo,foo] = mkdir(outdir);
write_data_set( outname , train_data , train_labels , test_data , test_labels );
cd(outdir);
[ status results ] = unix( [ '~/for_scott/ClassificationPackage-2009/classifiers/DecisionTrees/c4.5 -f ' name ' -u >',outdir,'/first_result.txt']);
!sed -n '/Evaluation on test data/,$p' first_result.txt > ./testing_data_list.txt
[ status results ] = unix( ['grep ITEM testing_data_list.txt > ./item_itemnos.txt']);
!sed -e 's/ITEM/ /g' item_itemnos.txt > ./itemnos.txt
incorrect_predictions = load( 'itemnos.txt' );
test_labels( incorrect_predictions+1 ) = test_labels(incorrect_predictions+1 ) * -1;
cd(currdir);
end


