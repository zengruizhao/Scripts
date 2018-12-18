clear;clc
para.distrib = {'normal'};%kernel, normal, bad:mn, mvmn
para.prior = {'uniform'};%empirical, uniform
best = [231, 166, 102, 37, 167, 232];% ring_inner_6_2 erode  No_erode 5
% best = [231, 166, 102, 37, 101, 36];% ring_inner_6_3 5
% best = [231, 194, 44, 244, 101, 109];% ring_inner_4_2 5
% best = [1, 31, 2, 161, 66, 67];% ring_inner_6_3 3
% best = [1, 2, 67, 336, 32, 161];% ring_inner_4_2 3 * LDA
% best = [371, 326, 218, 238, 277, 153];% No_erode 4 * NBayes
% best = [87, 86, 152, 21, 217, 22];% erode 4 ** LDA
% best = [98, 108, 308, 220, 22, 44];% ring_inner_6_3 4 * LDA
phase = 2;
load ../Neuroendocrine/Feature_5_3d_erode_ring_inner_6_2_HaralickLaws.mat
load ../Neuroendocrine/label_5_2cm.mat
phase_5 = phase;
if phase==2
   phase_5 = 5; %5
end
data= feature_process_new_1(Feature, phase_5);
data = data(:, best);
data_training = data(logical(~label(:, 3)), :);%
data_testing = data(logical(label(:, 3)), :);%
label_ = label(:, 1)-1;
label_training = label_(logical(~label(:, 3)));%
label_testing = label_(logical(label(:, 3)));%
load ../Neuroendocrine/Feature_4_3d_erode_ring_inner_6_2_HaralickLaws.mat
load ../Neuroendocrine/label_4_2cm.mat
data1 = feature_process_new_1(Feature, phase);
data1 = data1(:, best);
data1_training = data1(logical(~label(:, 3)), :);%
data1_testing = data1(logical(label(:, 3)), :);%
label_1 = label(:, 1)-1;
label_1_training = label_1(logical(~label(:, 3)));%
label_1_testing = label_1(logical(label(:, 3)));%
training_set = [data_training;data1_training];
testing_set = [data_testing;data1_testing];
All_data = [training_set;testing_set];
label = [label_;label_1];
training_labels = [label_training;label_1_training];
testing_labels = [label_testing;label_1_testing];
%%
training_set = svm_scale(training_set);
testing_set = svm_scale(testing_set);
%
para.classifier_name = 'LDA';% 'NBayes' 'LDA'
%%
[labels_pred, probability_pred, result, probability_pred_training]...
    = Lgenerate_Predicted_Label_of_test_data_no_FS(training_set, testing_set, training_labels, testing_labels, para);
result.training.auc
result.testing.auc
%% plot ROC
fprTrain = result.training.FPR;
tprTrain = result.training.TPR;
fprTest = result.testing.FPR;
tprTest = result.testing.TPR;
plot(fprTrain, tprTrain);
hold on;
plot(fprTest, tprTest);
%% Plot classifier
% one = training_set(training_labels==1, :);
% two = training_set(training_labels==0, :);
% % one = tsne(one, 'Algorithm','exact','NumDimensions',3);
% % two = tsne(two, 'NumDimensions',3);
% one = pca(one);
% two = pca(two);
% all_data = [one;two];
% % all_label = [ones(size(one, 1), 1);zeros(size(two, 1), 1)];
% % scatter3(one(:, 1), one(:, 2), one(:, 3), '*');
% % hold on
% % scatter3(two(:, 1), two(:, 2), two(:, 3), 'o');
% plot(one(:, 1), one(:, 2), '*');
% hold on;
% plot(two(:, 1), two(:, 2), 'o');
% hold off;