clear;clc;close all;
addpath(genpath('./script'));
addpath(genpath('/media/zzr/Data/git/code/trunk/matlab/general'));
addpath(genpath('/media/zzr/Data/git/code/trunk/matlab/images'));
All_path = '/media/zzr/Data/Neuroendocrine/2D_NET_CT/';
subpath = dir(All_path);
tic;case_num=0;
for i=3:length(subpath)
    subpath2 = dir([All_path subpath(i).name]);
    image_file = subpath2(end-2).name;%end:v end-1:p end-2:C-
    subpath3 = dir([All_path subpath(i).name '/' image_file, '/' '*.nrrd']); 
    if ~isempty(subpath3)
        if isempty(strfind(subpath3(1).name, 'label')) %img
            img_path = [All_path subpath(i).name '/' image_file, '/', subpath3(1).name];
            label_path = [All_path subpath(i).name '/' image_file, '/', subpath3(2).name];
        else
            img_path = [All_path subpath(i).name '/' image_file, '/', subpath3(2).name];
            label_path = [All_path subpath(i).name '/' image_file, '/', subpath3(1).name];
        end
        [img, img_meta] = nrrdread(img_path);
        [label, label_meta] = nrrdread(label_path);
        [m,n,slice] = size(label);
        for num=1:slice %% select slice
            if sum(sum(label(:,:,num)))
                img_out = img(:,:,num);
                label_out = label(:,:,num);
                fprintf('%d: case%d %dth slice was selected\n',i-2,str2num(subpath(i).name),num)
                break;
            end
        end
        case_num = case_num+1;
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
fprintf('There are %d cases in total.\n',case_num);
% save('V', 'V_img', 'V_label');
toc;
% templatevolume = imread('./data/8 Body 3.0 C.png');
% inputvolume= imread('./data/8 Body 3.0 CE.png');
% [outputvolume,standardization_map] = int_stdn_landmarks(inputvolume,templatevolume);