clear;clc;
which = 4;
% img_path = ['/home/zzr/Data/ISIC/densenet/temp_own_dir_aug/' num2str(which) '/'];
img_path = ['/home/zzr/Data/logo/raw_img/' num2str(which) '/'];
img_dir = dir([img_path '*.jpg']);
rand = randperm(length(img_dir), length(img_dir));
train_img = ['/home/zzr/Data/logo/train_test/pre_aug/train/' num2str(which) '/'];
test_img = ['/home/zzr/Data/logo/train_test/pre_aug/test/' num2str(which) '/'];
fprintf('train_set:\n')
for i = 1:round(length(rand)*0.8) % train
    disp(i);
    img = imread([img_path img_dir(rand(i)).name]);
    imwrite(img, [train_img img_dir(rand(i)).name]);
end
fprintf('test_set:\n')
for i = round(length(rand)*0.8)+1 : length(rand) % test
    disp(i)
    img = imread([img_path img_dir(rand(i)).name]);
    imwrite(img, [test_img img_dir(rand(i)).name]);
end
