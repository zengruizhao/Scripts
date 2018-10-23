%% 生成训练特征样本
clc;clear all;close all;
addpath('..\cellclass\unit\');
str_folder(1).name='cell_c39\';
str_folder(2).name='cell_c-\';
str_folder(3).name='cell_c39\';
str_folder(4).name='cell_c-\';
string(1).name='traindata_c_p';
string(2).name='traindata_c_n';
string(3).name='testfeat_l_21';
string(4).name='testfeat_c_21';
for x=1:2
    image_folder=['D:\Program Files\Matlab\work\cellclass\mat2\',str_folder(x).name];
    addpath(image_folder);
    file_ext = '.jpg';
    name_folder=image_folder;  %选择文件夹
    folder_content = dir ([name_folder,'*',file_ext]);  %文件夹内容
    n_folder_content = size (folder_content,1);  %文件夹内文件数量
    for featchoice=3:3
        features=[];
        for k=1:n_folder_content
            strin = [name_folder,folder_content(k,1).name];  
            image = imread(strin);  
            feature=choicefeat(image,featchoice);
            features =[features; feature];      
        end
        strf=string(x).name;
        choicefolder(strf,featchoice,features);    
    end
end
