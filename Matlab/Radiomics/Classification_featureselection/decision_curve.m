clear;clc
best = [87, 86, 152, 21, 217, 22];
phase = 4;
load ../Neuroendocrine/CT/Feature_5_3d_erode_HaralickLaws.mat
load ../Neuroendocrine/CT/label_5_2cm.mat
phase_5 = phase;
if phase==2
   phase_5 = 5; %5
end
data= feature_process_new_1(Feature, phase_5);
data = data(:, best);
data_training = data(logical(~label(:, 3)), :);%
data_testing = data(logical(label(:, 3)), :);%
data_testing_2cm = data(logical(label(:, 3))&logical(label(:, 4)), :);
label_ = label(:, 1)-1;
ki67 = label(:, 2);
label_training = label_(logical(~label(:, 3)));%
label_testing = label_(logical(label(:, 3)));%
label_testing_2cm = label_(logical(label(:, 3))&logical(label(:, 4)));%
ki67_training = ki67(logical(~label(:, 3)));
ki67_testing = ki67(logical(label(:, 3)));
ki67_2cm = ki67(logical(label(:, 3))&logical(label(:, 4)));
load ../Neuroendocrine/CT/Feature_4_3d_erode_HaralickLaws.mat
load ../Neuroendocrine/CT/label_4_2cm.mat
data1 = feature_process_new_1(Feature, phase);
data1 = data1(:, best);
data1_training = data1(logical(~label(:, 3)), :);%
data1_testing = data1(logical(label(:, 3)), :);%
data1_testing_2cm = data1(logical(label(:, 3))&logical(label(:, 4)), :);
label_1 = label(:, 1)-1;
ki67_1 = label(:, 2);
label_1_training = label_1(logical(~label(:, 3)));%
label_1_testing = label_1(logical(label(:, 3)));%
label_1_testing_2cm = label_1(logical(label(:, 3))&logical(label(:, 4)));%
ki67_1_testing = ki67_1(logical(label(:, 3)));
ki67_1_training = ki67_1(logical(~label(:, 3)));
ki67_1_2cm = ki67_1(logical(label(:, 3))&logical(label(:, 4)));
training_set = [data_training;data1_training];
testing_set = [data_testing;data1_testing];
testing_set_2cm = [data_testing_2cm;data1_testing_2cm];
label = [label_;label_1];
training_labels = [label_training;label_1_training];
testing_labels = [label_testing;label_1_testing];
KI67_testing = [ki67_testing;ki67_1_testing];
KI67_training = [ki67_training;ki67_1_training];
KI67_2cm = [ki67_2cm;ki67_1_2cm];
labels_2cm = [label_testing_2cm;label_1_testing_2cm];
%%
training_set = svm_scale(training_set);
testing_set = svm_scale(testing_set);
testing_set_2cm = svm_scale(testing_set_2cm);
%
para.classifier_name = 'LDA';
%%
[~, probability_pred, result, probability_pred_training]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(training_set, testing_set, training_labels, testing_labels, para);
[~, probability_pred_2cm, result_2cm, ~]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(training_set, testing_set_2cm, training_labels, labels_2cm, para);
Radiomics_training = probability_pred_training;
Radiomics_testing = probability_pred;
Radiomics_2cm = probability_pred_2cm;
result.training.auc
result.testing.auc
result_2cm.testing.auc
%
DC_testing(:, 1) = testing_labels;
DC_testing(:, 2) = KI67_testing;
DC_testing(:, 3) = Radiomics_testing;

DC_2cm(:, 1) = labels_2cm;
DC_2cm(:, 2) = KI67_2cm;
DC_2cm(:, 3) = probability_pred_2cm;
%
DC_training(:, 1) = training_labels;
DC_training(:, 2) = KI67_training;
DC_training(:, 3) = Radiomics_training;
xlswrite('DC_testing.xlsx', DC_testing);
xlswrite('DC_2cm.xlsx', DC_2cm);
xlswrite('DC_training.xlsx', DC_training);


