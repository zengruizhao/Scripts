clear;clc;
path = '/home/zzr/Data/IDRiD/New/EX/Data/aug/test_mask_visualization/';
outpath = '/home/zzr/Data/IDRiD/New/EX/Data/aug/test_mask_coco/';
dir = dir([path '*.png']);
for i = 1:length(dir)
    disp(i);
    image = imread([path dir(i).name]);
%     imwrite(image, [outpath dir(i).name(1:end-10) dir(i).name(13:end)]);
    out_name = [outpath dir(i).name(1:8) dir(i).name(12:end-4) dir(i).name(9:11) '_0.png'];
    imwrite(image, out_name);
end