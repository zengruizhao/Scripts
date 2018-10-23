clc;clear all;close all;
image_folder='E:\实验数据\oncotypeDX\提取小块\cell_r39\';
save_folder='E:\实验数据\oncotypeDX\提取小块\cell_r23\';
image_ext='.jpg';
folder_content=dir([image_folder, '*' ,image_ext]);
number_content=size(folder_content,1);
for i=1:number_content
    image_name=folder_content(i,1).name;
    image=imread([image_folder image_name]);
    image_resize=imresize(image,[23 23]);
    imwrite(image_resize,[save_folder num2str(i) image_ext]);
end  