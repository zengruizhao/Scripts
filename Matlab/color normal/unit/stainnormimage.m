% ͼ���׼��
tic;
clc;clear all;close all;
addpath('..\cellclass\unit\');
image_folder='D:\����\training\training\';
save_folder='D:\����\training\train_normal\';
target=imread('D:\Program Files\Matlab\work\code_Cell_detect_SSAE\target2.jpg');
% target=imread('..\cellclass\traindata2\image_l\SGOT-005-d_ANB.tif');
file_ext = '.tif';
name_folder=image_folder;  %ѡ���ļ���
folder_content = dir ([name_folder,'*',file_ext]);  %�ļ�������
n_folder_content = size (folder_content,1);  %�ļ������ļ�����
for k=1:n_folder_content
    fprintf(' Img %d/%d to be detected...\n ', k,n_folder_content); 
    toc
    string = [name_folder,folder_content(k,1).name];
    image = imread(string);    
    norm_img=stainnorm_reinhard(image,target);
    imwrite(norm_img,[save_folder,folder_content(k).name]);
end