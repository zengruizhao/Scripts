clear;clc;close all;
addpath(genpath('./script'));
addpath(genpath('E:\git\code\trunk\matlab\general'));
img_path = 'E:\Neuroendocrine\2D_NET_CT\15\C-\3 C-  3.0  B30f.nrrd';
label_path = 'E:\Neuroendocrine\2D_NET_CT\15\C-\6 C1  3.0  B30f-label.nrrd';
[img, img_meta] = nrrdread(img_path);
[label, label_meta] = nrrdread(label_path);
[m,n,slice] = size(label);
for num=1:slice
   if sum(sum(label(:,:,num)))
       img_out = img(:,:,num);
       label_out = label(:,:,num);
       break;
   end
end
% templatevolume = imread('./data/8 Body 3.0 C.png');
% inputvolume= imread('./data/8 Body 3.0 CE.png');
% [outputvolume,standardization_map] = int_stdn_landmarks(inputvolume,templatevolume);