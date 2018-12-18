function decision_tree_test
clear all
warning off
datasets = [1]; %choose datasets
dimensions = [4]; %choose dimensions
methods = [5]; %choose methods
krange = [2]; %choose range of k values for Isomap and LLE
%load bigtable

for dataset = datasets
    %[X, labels, titlestring] = loadthis(dataset); %open dataset
    cd DrKakadiarisFiles
    load TrainingData.mat
    load TestingData.mat
    labels = TrainingData(:,size(TrainingData',1));
    train = TrainingData;
    test = TestingData;
    X = TrainingData(:,1:size(TrainingData',1)-1);
    Ltrain = length(X);
    Ltest = length(TestingData);
    % X(Ltrain+1:(Ltrain+Ltest),:)=TestingData;
    titlestring = 'Kakadiaris';
    cd ../

    %diary off %%%turn diary off
    %class1=length(find(labels ==1));  class2=length(find(labels ==2));
    class1=length(find(labels >0));  class2=length(find(labels <=0));
    fprintf('\n %s \n Positives: %i Negatives: %i Ratio: %f \n', titlestring,class1,class2,(class1/(class1+class2)));

    if krange ==1
        krange = [3:(length(labels)*2/3-1)]; %run for what range of k values for Isomap/LLE
    end

    for newdim = dimensions
        dims=[1:newdim]; %dimensions
        fprintf('Dimensions 1 through %i \n', newdim);
        for method = methods
            [temp,methodstring] = Dimred(method, X, newdim, labels,1);
            if method ==5 | method ==6
                %run for several k values
                for k = krange
                    [temp, methodstring] = Dimred(method, X, newdim, labels,k);
                    [tp, tn, fp, fn] = decision_tree(temp, labels, titlestring,Ltrain,Ltest);
                    acc = (tp+tn)/(tp+tn+fp+fn);
                    acc_vec(1,k) =  acc;
                end
                [acc,bestk] = max(acc_vec);
                fprintf('%f %s, k = %s\n', acc,methodstring, num2str(bestk));
            else
                [temp, methodstring] = Dimred(method, X, newdim, labels,1);
                [tp, tn, fp, fn] = decision_tree(temp, labels, titlestring,Ltrain,Ltest);
                acc = (tp+tn)/(tp+tn+fp+fn);
                fprintf('%f %s\n', acc,methodstring);
            end
            %bigtable(dataset, newdim, method) = acc;
            unreducedDT(dataset) = acc;

        end
    end
    save('unreducedDT.mat','unreducedDT');
end
diary off

function [tp, tn, fp, fn] = decision_tree(data_set, data_labels, name,Ltrain,Ltest)
%formatting
name = 'Prostatecancertest'; % Don't use spaces in your names
if all(abs(data_labels)-1)==0
    %data_labels = (((data_labels-1)*2)-1)'; %transforms data labels 1,2 to -1,1

    % 1. First, we acquire the training set, training labels, testing set and testing labels.
    %    For this, we will divide our data set in two. We will find positive and negative
    %    examples to create a balanced training set.

    a = find( data_labels >  0 );
    b = find( data_labels <= 0 );

    % a_values = ceil(rand(1,ceil(size(a,1)/4))*size(a,1)) ;
    % b_values = ceil(rand(1,ceil(size(b,1)/4))*size(b,1))  ;
    a_values = 1:(size(a,1)/16) ;
    b_values = 1:(size(b,1)/16) ;

    % select training set
    training_set    = data_set( [ a(a_values) ; b(b_values) ] , : );
    training_labels = data_labels( [ a(a_values) ; b(b_values) ] , : );

    % select testing and then eliminate the training examples
    testing_set = data_set;
    testing_labels = data_labels;

    testing_set([ a(a_values) ; b(b_values) ] , : ) = [];
    testing_labels([ a(a_values) ; b(b_values) ] , : ) = [];
else
    %%training and testing data already split
    size(data_set)
    training_set= data_set(1:Ltrain,:);
    testing_set= data_set(Ltrain+1:(Ltrain+Ltest),:);
    
    a = find( data_labels >  0 );
    b = find( data_labels <= 0 );

    % a_values = ceil(rand(1,ceil(size(a,1)/4))*size(a,1)) ;
    % b_values = ceil(rand(1,ceil(size(b,1)/4))*size(b,1))  ;
    a_values = 1:(size(a,1)) ;
    b_values = 1:(size(b,1)) ;

    % select training set
    training_set    = data_set( [ a(a_values) ; b(b_values) ] , : );
    training_labels = data_labels( [ a(a_values) ; b(b_values) ] , : );

    % select testing and then eliminate the training examples
    testing_set = data_set;
    testing_labels = data_labels;

    testing_set([ a(a_values) ; b(b_values) ] , : ) = [];
    testing_labels([ a(a_values) ; b(b_values) ] , : ) = [];
end

test_labels = run_decision_tree2( name , training_set , training_labels , testing_set , testing_labels );
[tp, tn, fp, fn] = count_values(test_labels, testing_labels);