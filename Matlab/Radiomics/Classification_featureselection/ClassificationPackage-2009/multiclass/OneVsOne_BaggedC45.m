function [methodstring stats] = OneVsOne_BaggedC45(training_set , testing_set, training_labels, testing_labels)

methodstring = '1v1 bagged c4.5 classifier';

training_labels = training_labels(:);
testing_labels = testing_labels(:);

% recast all labels to 1,2,3,etc.
unique_labels = unique([training_labels; testing_labels]);
for i=1:numel(unique_labels)
    testing_labels(ismember(testing_labels,unique_labels(i))) = i;
    training_labels(ismember(training_labels,unique_labels(i))) = i;
end

% generate all 1v1 combinations
all_perms = sortrows(nchoosek(1:numel(unique_labels)));

for i=1:size(all_perms,1)
    class1 = all_perms(i,1);
    class2 = all_perms(i,2);
    
    % use 2 classes for training
    TRAIN_SET = training_set(ismember(training_labels,[class1 class2]),:);
    TRAIN_LABEL = training_labels(ismember(training_labels,[class1 class2]));
    
    % classify and generate decision
    stats(i) = Classify('BaggedC45',TRAIN_SET , testing_set, TRAIN_LABEL, testing_labels);
    temp = stats(i).prediction >= stats(i).threshold;
    stats(i).decision(temp) = unique_labels(class2);
    stats(i).decision(~temp) = unique_labels(class1);
    stats(i).classes = unique_labels(all_perms(i,:)); % output original classes so user can figure out which classes were used for training in each run
end
