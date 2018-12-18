function [methodstring,tp,tn,fp,fn,prediction] = C45(training_set , testing_set, training_labels, testing_labels)

methodstring = 'c4.5 classifier ';
[prediction] = c45_demo(training_set, testing_set, training_labels, testing_labels);                
[tp,tn,fp,fn]= count_values(testing_labels, prediction);