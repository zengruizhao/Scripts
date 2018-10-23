clear;clc;
img_path = '/home/zzr/Data/IDRiD/T1/SE/Data/pre_aug/temp/img/';
mask_path = '/home/zzr/Data/IDRiD/T1/SE/Data/pre_aug/temp/mask/';
img_dir = dir([img_path '*.png']);
rand = randperm(length(img_dir), length(img_dir));
train_img = '/home/zzr/Data/IDRiD/T1/SE/Data/pre_aug/train_test/train_img/';
test_img = '/home/zzr/Data/IDRiD/T1/SE/Data/pre_aug/train_test/test_img/';
train_mask = '/home/zzr/Data/IDRiD/T1/SE/Data/pre_aug/train_test/train_mask/';
test_mask = '/home/zzr/Data/IDRiD/T1/SE/Data/pre_aug/train_test/test_mask/';
fprintf('train_set:\n')
for i = 1:round(length(rand)*0.8) % train
    disp(i);
%     mask_temp = regMAp([img_path img_dir(rand(i)).name], '_', 'split');
%     mask_name = [mask_path 'IDRiD_' mask_temp{2} '_OD_' mask_temp{3} '_' mask_temp{4}];
%     mask_name_new = [mask_path 'IDRiD_' mask_temp{2} '_' mask_temp{3} '_' mask_temp{4}];
    mask_name = [mask_path img_dir(rand(i)).name(1:end-4)  '_SE.png'];
    img = imread([img_path img_dir(rand(i)).name]);
    mask = imread(mask_name);
    imwrite(img, [train_img img_dir(rand(i)).name]);
    imwrite(mask, [train_mask img_dir(rand(i)).name(1:end-4)  '.png']);
end
fprintf('test_set:\n')
for i = round(length(rand)*0.8)+1 : length(rand) % test
    disp(i)
%     mask_temp = regMAp([img_path img_dir(rand(i)).name], '_', 'split');
%     mask_name = [mask_path 'IDRiD_' mask_temp{2} '_OD_' mask_temp{3} '_' mask_temp{4}];
%     mask_name_new = [mask_path 'IDRiD_' mask_temp{2} '_' mask_temp{3} '_' mask_temp{4}];
     mask_name = [mask_path img_dir(rand(i)).name(1:end-4)  '_SE.png'];
    img = imread([img_path img_dir(rand(i)).name]);
    mask = imread(mask_name);
    imwrite(img, [test_img img_dir(rand(i)).name]);
     imwrite(mask, [test_mask img_dir(rand(i)).name(1:end-4)  '.png']);
end
