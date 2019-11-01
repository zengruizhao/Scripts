% clear;clc;
% img = imread('/media/zzr/Data/skin_xml/original_new/RawImage/original_img.tif');
for i=1:72:size(img, 1)-288
    fprintf('%d \\ %d\n', i, size(img, 1));
    for j=1:72:size(img, 2)-288
        patch = img(i:i+287, j:j+287, :);
        patch = imresize(patch, [144, 144]);
        imwrite(patch, ['/media/zzr/Data/skin_xml/original_new/tif/' num2str((j-1)/2) '_' num2str((i-1)/2) '.jpg']);
    end
end