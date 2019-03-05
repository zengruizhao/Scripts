clear;clc;close all;
addpath(genpath('./script'))
addpath(genpath('/media/zzr/Data/git/code/trunk/matlab/general'));
addpath(genpath('/media/zzr/Data/git/code/trunk/matlab/images'));
load '/media/zzr/My Passport/430/MRI/T2InformationPreprocess.mat';
path = '/media/zzr/My Passport/430/MRI/Resample/';
outpath = '/media/zzr/My Passport/430/MRI/IntensityStandardization/';
case_all = dir(path);
%% Raw Histogram 
% for i=3:length(case_all)
%     disp(i-2);
%     case_path = [path case_all(i).name];
%     color = 'b';
%     for ii=1:length(Information)
%        if contains(case_all(i).name, Information(ii, 1))
%            if contains(Information(ii, 5), 'GE'), color='r';
%            else, color='g';end
%        end
%     end
%     if color == 'b', fprintf(case_all(i).name);end
%     all_file = dir(case_path);
%     for j=3:length(all_file)
%         if contains(all_file(j).name, 'img') && contains(all_file(j).name, 'T2')
%             info = niftiinfo([case_path '/' all_file(j).name]);
%             V = niftiread(info);
%             out = round(rescale_range(V,0,2000));
%             inputdata=sort(out(:));
%             plotdist(inputdata, color);
%             title('Histograms'); hold on
%         end
%     end
% end
%% After Standardization
options.docheck = false;
options.rescaleMax = 2000;
options.zeroval = 0;
templatevolume = [];templatemask = [];
idx = randperm(length(case_all)-2, 2);
for ii=1:length(idx)
    case_path = [path case_all(idx(ii) + 2).name];
    all_file = dir(case_path);
    for j=3:length(all_file)
        if contains(all_file(j).name, 'img') && contains(all_file(j).name, 'T2')
            infoImg = niftiinfo([case_path '/' all_file(j).name]);
            Img = niftiread(infoImg);
            infoMask = niftiinfo([case_path '/' all_file(j).name(1:end-11) '_label.nii.gz']);
            Mask = niftiread(infoMask);
            templatevolume(end+1:end+length(Img(:))) = Img(:);
            templatemask(end+1:end+length(Img(:))) = Mask(:);
        end
    end
end
checkifInt = find(templatevolume); checkifInt = templatevolume(checkifInt(1));
if ceil(checkifInt) ~= floor(checkifInt),templatevolume = round(templatevolume);end
templatevolume = round(rescale_range(templatevolume, 0, options.rescaleMax));
templatedata=sort(templatevolume(:));
templatedata(templatedata <= options.zeroval) = [];
% 
templdata_forlm = templatevolume; templdata_forlm(logical(templatemask))=[];
templdata_forlm(templdata_forlm <= options.zeroval) = [];
if length(find(size(templdata_forlm)-1)) > 1, templdata_forlm = templdata_forlm(:);end
templvol_lm = prctile(templdata_forlm, (10:10:90));
% 
for i=3:length(case_all)
    disp(i-2);
    case_path = [path case_all(i).name];
    color = 'b';
    for ii=1:length(Information)
       if contains(case_all(i).name, Information(ii, 1))
           if contains(Information(ii, 5), 'GE'), color='r';
           else, color='g';end
       end
    end
    if color == 'b', fprintf(case_all(i).name);end
    all_file = dir(case_path);
    for j=3:length(all_file)
        if contains(all_file(j).name, 'img') && contains(all_file(j).name, 'T2')
            infoImg = niftiinfo([case_path '/' all_file(j).name]);
            V = niftiread(infoImg);
            infoMask = niftiinfo([case_path '/' all_file(j).name(1:end-11) '_label.nii.gz']);
            options.INCANCERMASKS = niftiread(infoMask);
            [outputvolume, ~] = int_stdn_landmarks_multiTemplate(V, templatedata, templvol_lm, options);
            plotdist(outputvolume, color);hold on;
            if ~exist([outpath case_all(i).name], 'dir'),mkdir([outpath case_all(i).name]);end
            img = outputvolume;mask = options.INCANCERMASKS;
            save([outpath case_all(i).name '/' all_file(j).name(1:end-11)], 'img', 'mask');
%             niftiwrite(outputvolume, [outpath case_all(i).name '/' all_file(j).name], infoImg);
        end
    end
end
title('Histograms'); hold off;




