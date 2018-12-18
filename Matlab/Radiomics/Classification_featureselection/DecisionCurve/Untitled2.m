clear;clc;close all;
feature = 2;
% training =csvread('../PNET_CT/DC_training1.csv');
% label_training = training(:, 1);
% x_training = training(:, feature);
% DECISIONCURVE(x_training', label_training');
%
figure
testing = csvread('../PNET_CT/DC_testing1.csv');
label_testing = testing(:, 1);
x_testing = testing(:, feature);
DECISIONCURVE(x_testing', label_testing');

figure
testing = csvread('../PNET_CT/DC_2cm1.csv');
label_testing = testing(:, 1);
x_testing = testing(:, feature);
DECISIONCURVE(x_testing', label_testing');
%% logistic
% mdl = fitglm(x_testing, label_testing, 'quadratic');
% result = predict(mdl, x_testing);
%% svm
% mdl = fitcsvm(x_training, label_training, 'KernelFunction', 'rbf');
% mdl = fitPosterior(mdl, x_training, label_training);
% [~, result] = predict(mdl, x_testing);
% result = result(:, 2);
%% LDA
% mdl = fitcdiscr(x_training, label_training);
% [~, result] = predict(mdl, x_testing);
% result = result(:, 2);
%%