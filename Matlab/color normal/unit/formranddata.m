% 生成随机块代特征样本
clc;clear all;close all;
addpath('..\cellclass\unit\');
image_folder='..\cellclass\traindata2\img_rand\';
addpath(image_folder);
file_ext = '.tif';
name_folder=image_folder;  %选择文件夹
folder_content = dir ([name_folder,'*',file_ext]);  %文件夹内容
n_folder_content = size (folder_content,1);  %文件夹内文件数量
randnum=600; %随机取块数量
wind=[10,19];

for featchoice=2:4
  for i=2:2
    features=[];
    windsize=wind(i);
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
            patch=image(randy(i)-windsize:randy(i)+windsize,randx(i)-windsize:randx(i)+windsize,:);  %获取每一个小块 
            feature=choicefeat(patch,featchoice); 
            features =[features; feature]; 
        end
    end
    windsize=2*windsize+1;
    string=['randfeat_',num2str(windsize)];
    choicefolder(string,featchoice,features);
  end
end
