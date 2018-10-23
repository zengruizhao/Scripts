%细胞检测，获取所有检测点
clc;clear all;close all;
addpath('..\cellclass\unit\');
save_folder='D:\Program Files\Matlab\work\cellclass\dots\l_groundtrue\';
image_folder='C:\Users\MrGong\Desktop\标准化全\';
image_ext = '.tif';
folder_content = dir ([image_folder,'*',image_ext]);  %image文件夹内容
n_folder_content = size (folder_content,1);  %文件夹内文件数量,即所需提取特征的图片数量

for k=1:n_folder_content
    name=folder_content(k,1).name;
    ind=strfind(name,'.tif');
    name=name(1:ind(1)-1);    
    string_img = ['C:\Users\MrGong\Desktop\label_l\' name '_BC_LI_l.tif'];
    image = imread(string_img); 
    cell=image(:,:,[1 2 3]);
%     [celldots] = cellocationc(cell);     
    [celldots] = cellocationl(cell);     
    save([save_folder name '.mat'],'celldots');
end
