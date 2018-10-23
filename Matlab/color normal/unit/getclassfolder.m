% SVM分类测试
clc;clear all;close all;
addpath('..\cellclass\unit\');
save_folder='D:\Program Files\Matlab\work\cellclass\fengji\typefolder\';
class_1_folder='D:\Program Files\Matlab\work\cellclass\fengji\typefolder\image\1\';%15 8+7
class_2_folder='D:\Program Files\Matlab\work\cellclass\fengji\typefolder\image\2\';%16 8+8
class_3_folder='D:\Program Files\Matlab\work\cellclass\fengji\typefolder\image\3\';%18 9+9
% need_folder=class_1_folder;
file_ext = '.tif';
folder_content = dir ([class_1_folder,'*',file_ext]);  %image文件夹内容
n_folder_content = size (folder_content,1);  
for k=1:n_folder_content
    name=folder_content(k,1).name;
    ind=strfind(name,'.tif');
    name=name(1:ind(1)-1);    
    class1folder(k,:)=name;
end
save([save_folder 'class1folder.mat'],'class1folder');

folder_content = dir ([class_2_folder,'*',file_ext]);  %image文件夹内容
n_folder_content = size (folder_content,1);  
for k=1:n_folder_content
    name=folder_content(k,1).name;
    ind=strfind(name,'.tif');
    name=name(1:ind(1)-1);    
    class2folder(k,:)=name;
end
save([save_folder 'class2folder.mat'],'class2folder');

folder_content = dir ([class_3_folder,'*',file_ext]);  %image文件夹内容
n_folder_content = size (folder_content,1);  
for k=1:n_folder_content
    name=folder_content(k,1).name;
    ind=strfind(name,'.tif');
    name=name(1:ind(1)-1);    
    class3folder(k,:)=name;
end
save([save_folder 'class3folder.mat'],'class3folder');
    