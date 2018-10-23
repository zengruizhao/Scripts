clear;clc;close all;
addpath(genpath('./script'));
addpath(genpath('E:\git\code\trunk\matlab\general'));
All_path = 'E:\Neuroendocrine\2D_NET_CT\';
addpath(genpath('E:\git\code\trunk\matlab\images'));
subpath = dir(All_path);
tic;case_num=0;
for i=6:7
    subpath2 = dir([All_path subpath(i).name]);
    image_file = subpath2(end-2).name;%end:v end-1:p end-2:C-
    subpath3 = dir([All_path subpath(i).name '\' image_file, '\' '*.nrrd']); 
    if ~isempty(subpath3)
        if isempty(strfind(subpath3(1).name, 'label')) %img
            img_path = [All_path subpath(i).name '\' image_file, '\', subpath3(1).name];
            label_path = [All_path subpath(i).name '\' image_file, '\', subpath3(2).name];
        else
            img_path = [All_path subpath(i).name '\' image_file, '\', subpath3(2).name];
            label_path = [All_path subpath(i).name '\' image_file, '\', subpath3(1).name];
        end
        [img, img_meta] = nrrdread(img_path);
        [label, label_meta] = nrrdread(label_path);
        [m,n,slice] = size(label);
        if i == 6
           img_out = img(:,:,61);
           label_out = label(:,:,21); 
        else
           img_out = img(:,:,37);
           label_out = label(:,:,31); 
        end
%% plot histograms
%         templatevolume = imread('./data/8 Body 3.0 C.png');
%         img_out = round(rescale_range(img_out,0,2000));% rescale
%         inputdata=sort(img_out(:));
%         plotdist(inputdata,'b');title('Histograms')
%         hold on;
%% save img & label selected
        V_img{str2num(subpath(i).name)} = img_out;
        V_label{str2num(subpath(i).name)} = label_out;
    end
end