clear;clc;
path = '/home/zzr/Data/ISIC/densenet/test/ISIC2018_Task3_Test_Input/';
out_path = '/home/zzr/Data/ISIC/densenet/test/';
dir_path = dir([path '*.jpg']);
for i = 1:length(dir_path)
    disp(i);
    img = imread([path dir_path(i).name]);
%     out =im2uint8(imresize(img, [528 528]))/255;% mask
    out = imresize(img, [224 224]); % image
    imwrite(out, [out_path dir_path(i).name(1:end-4) '.png']);
end