% ������������������
clc;clear all;close all;
addpath('..\cellclass\unit\');
image_folder='..\cellclass\traindata2\img_rand\';
addpath(image_folder);
file_ext = '.tif';
name_folder=image_folder;  %ѡ���ļ���
folder_content = dir ([name_folder,'*',file_ext]);  %�ļ�������
n_folder_content = size (folder_content,1);  %�ļ������ļ�����
randnum=600; %���ȡ������
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
            patch=image(randy(i)-windsize:randy(i)+windsize,randx(i)-windsize:randx(i)+windsize,:);  %��ȡÿһ��С�� 
            feature=choicefeat(patch,featchoice); 
            features =[features; feature]; 
        end
    end
    windsize=2*windsize+1;
    string=['randfeat_',num2str(windsize)];
    choicefolder(string,featchoice,features);
  end
end
