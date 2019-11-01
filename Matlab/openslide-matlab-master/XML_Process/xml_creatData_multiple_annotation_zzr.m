clear;clc;
addpath(genpath('../../openslide-matlab-master'));
addpath('./BOPointInPolygon')
ImgPath= '/run/user/1000/gvfs/afp-volume:host=Darwin-MI.local,user=zrzhao,volume=Data2/Skin/MF_10/Area Segmentation/';
xmlPath= '/run/user/1000/gvfs/afp-volume:host=Darwin-MI.local,user=zrzhao,volume=Data2/Skin/MF_10/Area Segmentation/';
xmlFile = dir([xmlPath,'*.xml']);
openslide_load_library();
%% Color edconding
Annotation = [255 % biao pi
              65280 % zhen pi
              16711680 % zhi fang
              8224125 % han xian
              16777215 % background
              16711935];% fair
% Idx = [2
%        1
%        2
%        4
%        1
%        2];
%%
% Idx = [2
%    1
%    2
%    8
%    0.5
%    4];
%% 20
Idx = [1
   1
   1
   1
   1
   1];
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
                x = str2double(temp.Children(z).Attributes(1).Value);
                y = str2double(temp.Children(z).Attributes(2).Value);
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
    fprintf('The number of levels: %d\n', numberOfLevels);
    for ii = 1:size(c, 1)% Annotation
        for kk=1:size(c{ii, 1}, 1)% Region
            idx = Idx(ii);
            fprintf('%dth annotation %dth egion, idx=%d...\n', ii, kk, idx);
            cell=c{ii, 1}{kk,1};
            xmin=max(min(cell2mat(cell(:,1))), 0);
            ymin=max(min(cell2mat(cell(:,2))), 0);
            xmax=min(max(cell2mat(cell(:,1))), Width);
            ymax=min(max(cell2mat(cell(:,2))), Height);
            h = xmax-xmin;
            w = ymax-ymin;
            height = 224;
            width = 224;
            %% gurrante extract one region at least
            if w<width || h<height
                h = max(height, h);
                w = max(width, w);
            end
            %%
            for i=1:(height/idx):(h-height+1)
                for j=1:(width/idx):(w-width+1)
                    pointx = xmin +  width/2 + j-1; %
                    pointy = ymin+  height/2 + i-1;%
                    point = [pointx,pointy];
                    stage = BOPointInPolygon(cell2mat(cell), point);
                    if (stage == 'i') %||stage== 'o'
                        xPose = xmin + j-1;
                        yPose = ymin + i-1;
                        if ((xPose + width - 1) < Width) && ((yPose + height - 1) < Height)
                            patch = openslide_read_region(slide, xmin + j - 1, ymin + i-1, width, height, 1);
                            II=patch(:,:,2:4);
                            JPath=['/media/zzr/SW/Skin_xml/WSI_20/',xmlFile(k).name(1:19), '/',num2str(ii)];
                            if ~exist(JPath)
                                mkdir(JPath);
                            end
                            JpgPath=[JPath, '/', xmlFile(k).name(1:19), '_', num2str(kk), '_', num2str(j), '_', num2str(i), '.png'];
                            imwrite(II,JpgPath);
                        end
                    end
                end      
            end
        end
    end
end
toc
