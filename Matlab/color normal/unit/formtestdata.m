% 生成测试用特征样本
clc;clear all;close all;
addpath('..\cellclass\unit\');
image_folder='..\cellclass\testdata2\image\';
label_c_folder='..\cellclass\testdata2\label_c\';
label_l_folder='..\cellclass\testdata2\label_l\';
addpath(image_folder);
addpath(label_c_folder);
addpath(label_l_folder);
image_ext = '.tif';
label_ext = '.tif';
folder_content = dir ([image_folder,'*',image_ext]);  %image文件夹内容
labelc_content = dir ([label_c_folder,'*',label_ext]);  %label_c文件夹内容
labell_content = dir ([label_l_folder,'*',label_ext]);  %label_l文件夹内容
n_folder_content = size (folder_content,1);  %文件夹内文件数量,即所需提取特征的图片数量
wind=[21,39];  
for featchoice=5:5
    for i=1:1
        windsize=wind(i);
        features_c=[];
        features_l=[];
        for k=1:size (folder_content,1)            
            string_img = [image_folder,folder_content(k,1).name];
            image = imread(string_img); 
            string_l = [label_l_folder,labell_content(k,1).name];
            image_l = imread(string_l);  
            string_c = [label_c_folder,labelc_content(k,1).name];
            image_c = imread(string_c);  
            [trainf_l trainf_c]=trainmatrix(image , image_l, image_c, windsize,featchoice);
            features_c =[features_c; trainf_c];
            features_l =[features_l; trainf_l];    
        end
        stringc=['testfeat_c_',num2str(windsize) ];
        eval([stringc '=features_c;']);
        str='..\cellclass\mat2\';
        switch featchoice
            case 1
                folder='hdgs\';
                save([str,folder,stringc],stringc);
            case 2
                folder='statxture\';
                save([str,folder,stringc],stringc);
            case 3
                folder='colorMoments\';
                save([str,folder,stringc],stringc); 
            case 4
                folder='lbp\';
                save([str,folder,stringc],stringc); 
            case 5
                folder='hog\';
                save([str,folder,stringc],stringc); 
        end    
        stringl=['testfeat_l_',num2str(windsize)];
        eval([stringl '=features_l;']);
        str='..\cellclass\mat2\';
        switch featchoice
            case 1
                folder='hdgs\';
                save([str,folder,stringl],stringl);
            case 2
                folder='statxture\';
                save([str,folder,stringl],stringl);
            case 3
                folder='colorMoments\';
                save([str,folder,stringl],stringl); 
            case 4
                folder='lbp\';
                save([str,folder,stringl],stringl); 
            case 5
                folder='hog\';
                save([str,folder,stringl],stringl); 
        end  
    end
end


 
