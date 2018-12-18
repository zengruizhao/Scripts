clear;clc
para.distrib = {'normal'};%kernel, normal, bad:mn, mvmn
para.prior = {'uniform'};%empirical, uniform
phase = 2;
load ../Neuroendocrine/Feature_5_3d_erode.mat
load ../Neuroendocrine/label_5.mat
phase_5 = phase;
if phase==2
   phase_5 = 5; %5
end
data= feature_process_new_1(Feature, phase_5);
data = data(:, [37, 101, 102, 36]);
data_training = data(logical(~label(:, 3)), :);
data_testing = data(logical(label(:, 3)), :);
label_ = label(:, 1)-1;
label_training = label_(logical(~label(:, 3)));
label_testing = label_(logical(label(:, 3)));
load ../Neuroendocrine/Feature_4_3d_erode.mat
load ../Neuroendocrine/label_4.mat
data1 = feature_process_new_1(Feature, phase);
data1 = data1(:, [37, 101, 102, 36]);
data1_training = data1(logical(~label(:, 3)), :);
data1_testing = data1(logical(label(:, 3)), :);
label_1 = label(:, 1)-1;
label_1_training = label_1(logical(~label(:, 3)));
label_1_testing = label_1(logical(label(:, 3)));
Data1_training = [data_training;data1_training];
Data1_testing = [data_testing;data1_testing];
training_labels = [label_training;label_1_training];
testing_labels = [label_testing;label_1_testing];
%
phase = 2;
load ../Neuroendocrine/Feature_5_3d_erode.mat
load ../Neuroendocrine/label_5.mat
phase_5 = phase;
if phase==2
   phase_5 = 2; %5
end
data= feature_process_new_1(Feature, phase_5);
data = data(:, 37);
data_training = data(logical(~label(:, 3)), :);
data_testing = data(logical(label(:, 3)), :);
load ../Neuroendocrine/Feature_4_3d_erode.mat
load ../Neuroendocrine/label_4.mat
data1 = feature_process_new_1(Feature, phase);
data1 = data1(:, 37);
data1_training = data1(logical(~label(:, 3)), :);
data1_testing = data1(logical(label(:, 3)), :);
Data2_training = [data_training;data1_training];
Data2_testing = [data_testing;data1_testing];
%
phase =3;
load ../Neuroendocrine/Feature_5_3d_erode_ring_inner.mat
load ../Neuroendocrine/label_5.mat
phase_5 = phase;
if phase==2
   phase_5 = 2; %5
end
data= feature_process_new_1(Feature, phase_5);
data = data(:, 36);
data_training = data(logical(~label(:, 3)), :);
data_testing = data(logical(label(:, 3)), :);
load ../Neuroendocrine/Feature_4_3d_erode_ring_inner.mat
load ../Neuroendocrine/label_4.mat
data1 = feature_process_new_1(Feature, phase);
data1 = data1(:, 36);
data1_training = data1(logical(~label(:, 3)), :);
data1_testing = data1(logical(label(:, 3)), :);
Data3_training = [data_training;data1_training];
Data3_testing = [data_testing;data1_testing];
training_set = [Data1_training, Data2_training, Data3_training];
testing_set = [Data1_testing, Data2_testing, Data3_testing];
training_set = svm_scale(training_set);
testing_set = svm_scale(testing_set);
feature_list = ones(1,size(training_set,2));
%
para.classifier_name = 'NBayes';% 'NBayes' 'LDA'
para.T_on_predicted_pro = 0.5;
%%
[labels_pred, probability_pred, result, probability_pred_training]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(training_set, testing_set, training_labels, testing_labels, para);
result.auc_train
result.auc_test