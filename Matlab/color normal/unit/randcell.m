%% ��������������
clc;clear all;close all;
addpath('..\cellclass\unit\');
image_folder='..\cellclass\traindata2\img_rand\';
addpath(image_folder);
file_ext = '.tif';
name_folder=image_folder;  %ѡ���ļ���
folder_content = dir ([name_folder,'*',file_ext]);  %�ļ�������
n_folder_content = size (folder_content,1);  %�ļ������ļ�����
randnum=400; %���ȡ������
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
        patch=image(randy(i)-windsize:randy(i)+windsize-1,randx(i)-windsize:randx(i)+windsize-1,:);  %��ȡÿһ��С�� 
        imwrite(patch,['D:\����\training\sn2\' num2str(sum) '.jpg']);
        sum=sum+1;
    end
end

