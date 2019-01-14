clear;clc;close all;
addpath(genpath('./script'));
addpath(genpath('E:\git\code\trunk\matlab\general'));
All_path = 'E:\Neuroendocrine\2D_NET_CT\';
addpath(genpath('E:\git\code\trunk\matlab\images'));
subpath = dir(All_path);
tic;case_num=0;
for i=3:length(subpath)
    subpath2 = dir([All_path subpath(i).name]);
    image_file = subpath2(end).name;%end:v end-1:p end-2:C-
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
        [~,~,slicee] = size(img);
        if slice ~= slicee;disp(str2num(subpath(i).name));end
        case_num = case_num+1;
    end
end
fprintf('There are %d cases in total.\n',case_num);
toc;
