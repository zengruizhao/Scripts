%% semantic for all labels
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
    imshow(MASK_filled, []);
    MASK_filled = MASK_filled - 1;
    MASK_filled(MASK_filled==-1) = 6;
    WIDTH = 512;
    HEIGHT = 512;
    stride = 64;
    %%
    for i=1:stride/mask_downsample:(size(MASK_filled, 2)-WIDTH/mask_downsample)
        for j=1:stride/mask_downsample:(size(MASK_filled, 1) - HEIGHT/mask_downsample)
            mask = MASK_filled(j:(j + HEIGHT/mask_downsample), i:(i+WIDTH/mask_downsample));
            if length(unique(mask)) == 1 || ~isempty(logical(unique(mask)==6)) && length(unique(mask)) == 2
                continue;
            elseif length(find(mask==6)) > (size(mask, 1) * size(mask, 2) / 8) % decrease unlabeled area as much as possible
                continue;
            else
                mask = imresize(mask, [HEIGHT, WIDTH], 'nearest');
                img = openslide_read_region(slide, i * mask_downsample, j * mask_downsample, WIDTH, HEIGHT, 1);
                II = img(:,:,2:4);
                imgpath=['/media/zzr/SW/Skin_xml/semantic_new/',xmlFile(k).name(1:19), '/img'];
                maskpath = ['/media/zzr/SW/Skin_xml/semantic_new/',xmlFile(k).name(1:19), '/mask'];
                if ~exist(imgpath, 'dir');mkdir(imgpath);end
                if ~exist(maskpath, 'dir');mkdir(maskpath);end
                imgpng = [imgpath, '/', xmlFile(k).name(1:19), '_', num2str(i), '_', num2str(j), '.png'];
                maskpng = [maskpath, '/', xmlFile(k).name(1:19), '_', num2str(i), '_', num2str(j), '.png'];
                imwrite(II, imgpng);  
                imwrite(uint8(mask), maskpng);  
            end
        end
    end
end
toc