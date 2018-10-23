clear;clc;close all;
load V;
addpath(genpath('./script'));
addpath(genpath('/media/zzr/Data/git/code/trunk/matlab/general'));
addpath(genpath('/media/zzr/Data/git/code/trunk/matlab/images'));
templatevolume = V_img{2};
inputvolume = V_img{2};
[outputvolume,standardization_map] = int_stdn_landmarks(inputvolume,templatevolume);