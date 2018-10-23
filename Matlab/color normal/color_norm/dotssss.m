clc;clear all;close all;
image_folder='E:\git\nomalpic\big\2/';
savepath = './output/2/';
image_ext = '.tif';
folder_content = dir ([image_folder,'*',image_ext]);
n_folder_content = size (folder_content,1); 
target=imread('./region_391masson(1,1)_NMF_C1.tif');
for k= 1: length(folder_content)
    t0 = clock;
    fprintf(' %d/%d to be Operated...\n ', k,n_folder_content);  
    string_img = [image_folder,folder_content(k,1).name];
%     name=folder_content(k,1).name;
%     ind=strfind(name,'.bmp');
%     name=name(1:ind(1)-1);
    image = imread(string_img); 
    image=stainnorm_reinhard(image,target);
    img=image(:,:,[1 2 3]);
    imwrite(img,[savepath folder_content(k,1).name]);
end

