% 细胞块精选
clc;clear all;close all;

yuanshi_folder='C:\Users\MrGong\Desktop\cellneed4\';
zhengque_folder='C:\Users\MrGong\Desktop\cellneed - 副本 - 副本\';
addpath(yuanshi_folder);
addpath(zhengque_folder);

image_ext = '.jpg';
yuanshi_content = dir ([yuanshi_folder,'*',image_ext]);  %c文件夹内容
zhengque_content = dir ([zhengque_folder,'*',image_ext]);  %l文件夹内容
n_folder = size (zhengque_content,1);  %文件夹内文件数量
n1_folder = size (yuanshi_folder,1);  %文件夹内文件数量
name=zeros(1,n_folder);
 for k=1:n_folder  
    name(k)=str2num(cell2mat(regexp(zhengque_content(k,1).name,'\d', 'match')));
 end
 rmpath(zhengque_folder);
 for k=1:n_folder  
%   if(name(k)<3872)
    string_img = [yuanshi_folder,num2str(name(k)),'.jpg'];
    image = imread(string_img); 
    imwrite(image,['C:\Users\MrGong\Desktop\cellneed44\' num2str(name(k)) '.jpg']);
%   end
 end