clear;clc;
load heart_scale.mat
% Split Data
train_data = heart_scale_inst(1:150,:);
train_label = heart_scale_label(1:150,:);
test_data = heart_scale_inst(151:270,:);
test_label = heart_scale_label(151:270,:);
% Linear Kernel
model_linear = trainsvm(train_label, train_data, '-t 0');
[predict_label_L, accuracy_L, dec_values_L] = predictsvm(test_label, test_data, model_linear);