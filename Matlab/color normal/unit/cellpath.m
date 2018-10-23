% ���ɲ�ͬ���ڳߴ�ϸ��С��
clc;clear all;close all;
addpath('..\cellclass\unit\');
%% �ò����γɲ�����trainfeature
% image_c_folder='..\cellclass\traindata2\image_c\';
% image_l_folder='..\cellclass\traindata2\image_l\';
% label_c_folder='..\cellclass\traindata2\label_c\';
% label_l_folder='..\cellclass\traindata2\label_l\';
% addpath(image_c_folder);
% addpath(image_l_folder);
% addpath(label_c_folder);
% addpath(label_l_folder);
image_l_folder='C:\Users\MrGong\Desktop\��׼��ȫ\';
label_l_folder='C:\Users\MrGong\Desktop\label_c\';


image_ext = '.tif';
label_ext = '.tif';
% folder_c_content = dir ([image_c_folder,'*',image_ext]);  %c�ļ�������
folder_l_content = dir ([image_l_folder,'*',image_ext]);  %l�ļ�������
% labelc_content = dir ([label_c_folder,'*',label_ext]);  %label_c�ļ�������
labell_content = dir ([label_l_folder,'*',label_ext]);  %label_l�ļ�������

% n_folder_c = size (folder_c_content,1);  %�ļ������ļ�����
n_folder_l = size (folder_l_content,1);  %�ļ������ļ�����
%% L
image_folder=image_l_folder;  folder_content=folder_l_content;
label_folder=label_l_folder;  label_content=labell_content;
% %% C
% image_folder=image_c_folder;  folder_content=folder_c_content;
% label_folder=label_c_folder;  label_content=labelc_content;

windsize=39;
sum=1;
kk=1;
  for k=1:size (folder_content,1)
        string_img = [image_folder,folder_content(k,1).name];
        image = imread(string_img); 
        string_lab = [label_folder,label_content(k,1).name];
        image_lab = imread(string_lab);  
        image_lab = image_lab(:,:,[1 2 3]);
        xy_lab=label_c(image_lab);  %��ǵ��������
        windin=floor(windsize/2); %С
        windout=ceil(windsize/2); %��
        [zb l n]=bwboundaries(xy_lab);% �ⲿ����   
        [m l z]=size(image);    
        for i=1:n
            z=zb(i,1);
            z=cell2mat(z);
            x1=z(:,2);  %������ͨ��������
            y1=z(:,1); 
            xy=[x1 y1];
            [cx cy A]=centroid(xy);  %cell��������
            if (rem(kk,1)==0)
                if((windout<cx&&cx<(l-windout))&&(windout<cy&&cy<(m-windout)))  %ȥ��ͼ��߽總����ϸ��
                    cell=image(cy-windin-1:cy+windin-1,cx-windin:cx+windin,:);  %ѡȡ���ڴ�С��ϸ������          
                    imwrite(cell,['C:\Users\MrGong\Desktop\cell_c39\' num2str(sum) '.jpg']);
                    sum=sum+1; %ͳ����������
                end 
            end
            kk=kk+1;
        end
   end



 
