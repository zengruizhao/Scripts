clear;clc;
train_img_path = '/home/zzr/Data/NKI/1_15/train/img/';
train_mask_path = '/home/zzr/Data/NKI/1_15/train/label/';
test_img_path = '/home/zzr/Data/NKI/1_15/test/img/';
test_mask_path = '/home/zzr/Data/NKI/1_15/test/label/';
%% train
fp = fopen([train_img_path(1:end-10) 'train.txt'], 'wt');
train_img_dir = dir([train_img_path '*.png']);
train_mask_dir = dir([train_mask_path '*.png']);
for i = 1:length(train_img_dir)
   disp(i);
   fprintf(fp, '%s\n', [train_img_path, train_img_dir(i).name, ' ', train_mask_path, train_mask_dir(i).name]);
end
%% test
fp = fopen([train_img_path(1:end-10) 'test.txt'], 'wt');
test_img_dir = dir([test_img_path '*.png']);
test_mask_dir = dir([test_mask_path '*.png']);
for i = 1:length(test_img_dir)
   disp(i);
   fprintf(fp, '%s\n', [test_img_path, test_img_dir(i).name, ' ', test_mask_path, test_mask_dir(i).name]);
end
