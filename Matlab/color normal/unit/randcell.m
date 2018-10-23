%% 生成随机块代样本
clc;clear all;close all;
addpath('..\cellclass\unit\');
image_folder='..\cellclass\traindata2\img_rand\';
addpath(image_folder);
file_ext = '.tif';
name_folder=image_folder;  %选择文件夹
folder_content = dir ([name_folder,'*',file_ext]);  %文件夹内容
n_folder_content = size (folder_content,1);  %文件夹内文件数量
randnum=400; %随机取块数量
wind=19;
sum=1;
windsize=wind;
for k=1:n_folder_content
    string = [name_folder,folder_content(k,1).name];
    image = imread(string);    
    [ysz,xsz,csz]=size(image);
    ysz=ysz-windsize;
    xsz=xsz-windsize;
    warning('off');
    randx=randint(1,randnum,[windsize+1,xsz]) ;
    randy=randint(1,randnum,[windsize+1,ysz]) ;
    warning('on');      
    for i=1:randnum
        patch=image(randy(i)-windsize:randy(i)+windsize-1,randx(i)-windsize:randx(i)+windsize-1,:);  %获取每一个小块 
        imwrite(patch,['D:\下载\training\sn2\' num2str(sum) '.jpg']);
        sum=sum+1;
    end
end

