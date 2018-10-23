% 生成不同窗口尺寸细胞小块
clc;clear all;close all;
addpath('..\cellclass\unit\');
%% 该部分形成测试用trainfeature
% image_c_folder='..\cellclass\traindata2\image_c\';
% image_l_folder='..\cellclass\traindata2\image_l\';
% label_c_folder='..\cellclass\traindata2\label_c\';
% label_l_folder='..\cellclass\traindata2\label_l\';
% addpath(image_c_folder);
% addpath(image_l_folder);
% addpath(label_c_folder);
% addpath(label_l_folder);
image_l_folder='C:\Users\MrGong\Desktop\标准化全\';
label_l_folder='C:\Users\MrGong\Desktop\label_c\';


image_ext = '.tif';
label_ext = '.tif';
% folder_c_content = dir ([image_c_folder,'*',image_ext]);  %c文件夹内容
folder_l_content = dir ([image_l_folder,'*',image_ext]);  %l文件夹内容
% labelc_content = dir ([label_c_folder,'*',label_ext]);  %label_c文件夹内容
labell_content = dir ([label_l_folder,'*',label_ext]);  %label_l文件夹内容

% n_folder_c = size (folder_c_content,1);  %文件夹内文件数量
n_folder_l = size (folder_l_content,1);  %文件夹内文件数量
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
        xy_lab=label_c(image_lab);  %标记点坐标矩阵
        windin=floor(windsize/2); %小
        windout=ceil(windsize/2); %大
        [zb l n]=bwboundaries(xy_lab);% 外部轮廓   
        [m l z]=size(image);    
        for i=1:n
            z=zb(i,1);
            z=cell2mat(z);
            x1=z(:,2);  %单个联通区域坐标
            y1=z(:,1); 
            xy=[x1 y1];
            [cx cy A]=centroid(xy);  %cell中心坐标
            if (rem(kk,1)==0)
                if((windout<cx&&cx<(l-windout))&&(windout<cy&&cy<(m-windout)))  %去除图像边界附近的细胞
                    cell=image(cy-windin-1:cy+windin-1,cx-windin:cx+windin,:);  %选取窗口大小的细胞区域          
                    imwrite(cell,['C:\Users\MrGong\Desktop\cell_c39\' num2str(sum) '.jpg']);
                    sum=sum+1; %统计区块数量
                end 
            end
            kk=kk+1;
        end
   end



 
