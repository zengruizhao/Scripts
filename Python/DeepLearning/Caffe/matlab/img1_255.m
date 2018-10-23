clear;clc;
path = '/home/zzr/Data/IDRiD/New/OD/Data/aug/train_mask/';
out_path = '/home/zzr/Data/IDRiD/New/OD/Data/aug/train_mask_visualization/';
dir_path = dir([path '*.png']);
for i = 1:length(dir_path)
    disp(i);
    img = imread([path dir_path(i).name])*255;
    imwrite(img, [out_path dir_path(i).name(1:end-4) '.png']);
end