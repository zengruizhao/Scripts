function prediction = c45_demo(training_set, testing_set, training_labels, testing_labels)
% Runs c4.5 Decision Trees

% format labels from (-1,1) to (1,2)
if training_labels < 0
    training_labels(training_labels==-1) = 2;
    testing_labels(testing_labels==-1) =2;
end

% format labels from (0,1) to (1,2)
if any(training_labels == 0)
    training_labels(training_labels==0) = 2;
    testing_labels(testing_labels==0) = 2;
end

name = 'auto';
currdir = pwd;
outdir = [currdir,'/dump/',name]; outname = [outdir,'/',name];
[foo,foo,foo] = mkdir(outdir);
write_data_set( outname , training_set , training_labels , testing_set , testing_labels );
cd(outdir);
[ status results ] = unix( ['/home/abasavan/for_scott/ClassificationPackage-2009/classifiers/DecisionTrees/c4.5 -f ' name ' -u >',outdir,'/first_result.txt']);
!sed -n '/Evaluation on test data/,$p' first_result.txt > ./testing_data_list.txt
[ status results ] = unix( ['grep ITEM testing_data_list.txt > ./item_itemnos.txt']);
!sed -e 's/ITEM/ /g' item_itemnos.txt > ./itemnos.txt
incorrect_predictions = load( 'itemnos.txt' );

% testing_labels = (testing_labels*2)-1; %format labels to -1 and 1
prediction = testing_labels;

prediction( incorrect_predictions+1 ) = testing_labels(incorrect_predictions+1 ) * -1; % flip labels for incorrect
cd(currdir);
