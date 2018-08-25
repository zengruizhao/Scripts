clear;clc;
path = './1/';
img_file = dir([path '*.tif']);
for i=1:length(img_file)
    disp(i);
    img = imread([path img_file(i).name]);
    vector{i} = colorMom_zzr(img);
end
