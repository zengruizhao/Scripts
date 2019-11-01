%% 2 classes: biao pi and zhenpi 
clear;clc;
addpath(genpath('../../openslide-matlab-master'));
addpath('./BOPointInPolygon')
ImgPath= '/run/user/1000/gvfs/smb-share:server=darwin-mi,share=data2/Skin/MF_10/Area Segmentation/Train/';
xmlPath= '/run/user/1000/gvfs/smb-share:server=darwin-mi,share=data2/Skin/MF_10/Area Segmentation/Train/';
xmlFile = dir([xmlPath,'*.xml']);
openslide_load_library();
%% Color edconding
Annotation = [255 % biao pi
              65280 % zhen pi
              16711680 % zhi fang
              8224125 % han xian
              16777215 % background
              16711935];% fair
downsample = 2;
mask_downsample = 8;
tic;
for k=1:length(xmlFile)% Xml file
    fprintf([xmlFile(k).name '...\n']);
    %% Read xml
    c={};
    oneXml = [xmlPath,xmlFile(k).name];
    xDoc = xmlread(oneXml);
    theStruct = parseChildNodes(xDoc);
    for i=2:size(theStruct.Children, 2)% Annotation
        annotation = str2double(theStruct.Children(i).Attributes(3).Value);
        index = find(Annotation == annotation);
        for j = 2:size(theStruct.Children(i).Children(2).Children, 2)% Region
            temp = theStruct.Children(i).Children(2).Children(j).Children(2);
            for z=1:size(temp.Children, 2)% Coordinate
                x = ceil(str2double(temp.Children(z).Attributes(1).Value)/downsample);
                y = ceil(str2double(temp.Children(z).Attributes(2).Value)/downsample);
                c{index, 1}{j-1, 1}{z, 1} = x;
                c{index, 1}{j-1, 1}{z, 2} = y;
            end
        end
    end
    %% Read Slide
    WSI = [ImgPath,xmlFile(k).name(1:end-4),'.ndpi'];
    slide = openslide_open(WSI);
    [~, ~, Width, Height, numberOfLevels, ...
    downsampleFactors, objectivePower] = openslide_get_slide_properties(slide);
    MASK_filled = zeros(ceil(Height/downsample/mask_downsample), ceil(Width/downsample/mask_downsample));
    for i = 1:size(c, 1)
        X=[];Y=[];
        MASK = zeros(ceil(Height/downsample/mask_downsample), ceil(Width/downsample/mask_downsample));
        for j = 1:size(c{i, 1}, 1)
            cell=c{i, 1}{j,1};
            for kk=1:length(cell)
               x = ceil(cell{kk, 1}/mask_downsample);
               if x<=0;x=1;end
               X(end+1) = x;
               y = ceil(cell{kk, 2}/mask_downsample);
               if y<=0;y=1;end
               Y(end+1) = y;
               MASK(x, y) = 1;
            end
            X(end+1) = X(1);
            Y(end+1) = Y(1);
            mask = poly2mask(X, Y, double(ceil(Height/downsample/mask_downsample)), double(ceil(Width/downsample/mask_downsample)));
            MASK_filled(mask==1) = i;
        end
    end
%     imshow(MASK_filled, []);
%     mask1 = imdilate(double(MASK_filled==1), strel('disk', 20));
%     union = mask1.*(MASK_filled==2);
    TEMP = MASK_filled;
%     MASK_filled(MASK_filled == 2) = 0;
%     MASK_filled(logical(union)) = 2;
%     TEMP(TEMP==2) = 0;
%%
    ROI_full_ind = find(MASK_filled==1);
    [ROI_full_row,ROI_full_col] = ind2sub(size(MASK_filled),ROI_full_ind);
    if min(ROI_full_col) / size(MASK_filled, 2) < 0.3 
        if min(ROI_full_row) / size(MASK_filled, 1) < 0.5 && max(ROI_full_row) / size(MASK_filled, 1) < 0.7
            for i=1:size(MASK_filled, 2)
                mask1 = find(MASK_filled(:,i) == 1);
                mask2 = find(MASK_filled(:,i) == 2);
                if ~isempty(mask1) && ~isempty(mask2)
                   MASK_filled(max(mask1):min(mask2), i) = 2; 
                end
            end
        elseif min(ROI_full_row) / size(MASK_filled, 1) > 0.3 && max(ROI_full_row) / size(MASK_filled, 1) > 0.5
            for i=1:size(MASK_filled, 2)
                mask1 = find(MASK_filled(:,i) == 1);
                mask2 = find(MASK_filled(:,i) == 2);
                if ~isempty(mask1) && ~isempty(mask2)
                   MASK_filled(max(mask2):min(mask1), i) = 2; 
                end
            end
        else
            for i=1:size(MASK_filled, 1)
                mask1 = find(MASK_filled(i, :) == 1);
                mask2 = find(MASK_filled(i, :) == 2);
                if ~isempty(mask1) && ~isempty(mask2)
                   MASK_filled(i, max(mask1):min(mask2)) = 2; 
                end
            end
        end
    else
        for i=1:size(MASK_filled, 1)
            mask1 = find(MASK_filled(i, :) == 1);
            mask2 = find(MASK_filled(i, :) == 2);
            if ~isempty(mask1) && ~isempty(mask2)
               MASK_filled(i, max(mask2):min(mask1)) = 2; 
            end
        end
    end
    %%
    for i=3:6
        MASK_filled(TEMP == i) = 0;
    end
    MASK_filled = MASK_filled - 1;
    MASK_filled(MASK_filled==-1) = 2;
%     imshow(MASK_filled, []);
%     pause
    WIDTH = 256;
    HEIGHT = 256;
    stride = 8;
    %
    bw_1 = bwboundaries(MASK_filled==0);
    for i = 1:length(bw_1) 
        if length(bw_1{i}) > 5
            coordinates = bw_1{i};
            for j = 1:stride:length(coordinates)
               x = coordinates(j, 2);
               y = coordinates(j, 1);
               if (x>HEIGHT/mask_downsample/2) && (y>WIDTH/mask_downsample/2)
                   mask = MASK_filled((y - HEIGHT/mask_downsample/2): (y + HEIGHT/mask_downsample/2), ...
                                      (x - WIDTH/mask_downsample/2): (x + WIDTH/mask_downsample/2));
                   if length(unique(mask)) == 1 || ~isempty(logical(unique(mask)==2)) && length(unique(mask)) == 2
                        continue;
                   elseif length(find(mask==2)) > (size(mask, 1) * size(mask, 2) / 4) % decrease unlabeled area as much as possible
                        continue;
                   else
                        mask = imresize(mask, [HEIGHT, WIDTH], 'nearest');
                        img = openslide_read_region(slide, (x - WIDTH/mask_downsample/2) * mask_downsample, ...
                                                    (y - HEIGHT/mask_downsample/2) * mask_downsample, WIDTH, HEIGHT, 1);
                        II = img(:,:,2:4);
                        imgpath=['/media/zzr/SW/Skin_xml/semantic_2/',xmlFile(k).name(1:19), '/img'];
                        maskpath = ['/media/zzr/SW/Skin_xml/semantic_2/',xmlFile(k).name(1:19), '/mask'];
                        if ~exist(imgpath, 'dir');mkdir(imgpath);end
                        if ~exist(maskpath, 'dir');mkdir(maskpath);end
                        imgpng = [imgpath, '/', xmlFile(k).name(1:19), '_', num2str(x), '_', num2str(y), '.png'];
                        maskpng = [maskpath, '/', xmlFile(k).name(1:19), '_', num2str(x), '_', num2str(y), '.png'];
                        imwrite(II, imgpng);  
                        imwrite(uint8(mask), maskpng);  
                   end
               end
            end
        end
    end
end
toc