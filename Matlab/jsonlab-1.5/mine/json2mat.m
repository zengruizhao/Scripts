clear;clc;
json_path= 'E:\Skin\ISIC2017\ISIC-2017_Training_Part2_GroundTruth';
img_path = 'E:\Skin\ISIC2017\ISIC-2017_Training_Data\';
save_path = 'new\';
json = dir(json_path);
parfor ii=1:4
    for i=3:size(json)
        disp(i-2);
        img = imread([img_path json(i).name(1:end-14) '_superpixels.png']);
        [m,n,~] = size(img);
        A = zeros(m,n);
        index = decodeSuperpixelIndex(img);
        [col_val,ind_first]=unique(index,'first');
        data =cell2mat(struct2cell(loadjson(json(i).name)));
        for j=0:size(col_val)-1
            [x,y] = find(index == j);%return the coordinate of specific pixel
            A(x,y) = data(ii,j+1);
        end
        for x=1:m
            for y=1:n
                if A(x,y) == 1
                    A(x,y) = 255;
                end
            end
        end
        imwrite(A,[save_path num2str(ii) '\' json(i).name(1:end-14) '.png']);
    end
end

