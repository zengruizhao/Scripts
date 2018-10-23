clear;
clc;
addpath('../../openslide-matlab');
addpath('./BOPointInPolygon');
ImgPath='/home/xjw/Nuclear_Seg_SegNet/Data/train/img/';
xmlPath='/home/xjw/Nuclear_Seg_SegNet/Data/train/xml/';
out_Path = '/home/xjw/Nuclear_Seg_SegNet/Data/train/mask2/';

xmlFile=dir([xmlPath,'*.xml']);
for k=1:length(xmlFile)
    oneXml=[xmlPath,xmlFile(k).name];
    c = parseCamelyon16XmlAnnotations_ccf(oneXml);
%     c = getAnnotation(oneXml);
    mask = uint8(zeros(1000, 1000, 3));
    mask_g = mask(:, :, 2);
    LL=size(c);
    LL=LL(1,1);
    for kk=1:LL
%         set (gcf, 'position', [0, 0, 0, 0]);
        cell = ceil(c{kk,1} * 1);
        x_axis = cell(:, 1);
        y_axis = cell(:, 2);
        x = find(x_axis < 1);
        y = find(y_axis < 1);
        if ~isempty(x)
            for nn = 1: length(x)
                x_axis(x(nn)) = 1;
            end
        end
        if ~isempty(y)
            for nn = 1: length(y)
                y_axis(y(nn)) = 1;
            end
        end
        x = find(x_axis > 1000);
        y = find(y_axis > 1000);
        if ~isempty(x)
            for nn = 1: length(x) 
                x_axis(x(nn)) = 1000;
            end
        end
        if ~isempty(y)
            for nn = 1: length(y)
                y_axis(y(nn)) = 1000;
            end
        end
        for kkk = 1:length(cell)
            mask_g(x_axis(kkk), y_axis(kkk)) = 255;
        end
%         axis(gca,'off');
%         plot(x_axis, y_axis);
%         hold on
%         if kk = 1
%             xx = x_axis;
%             yy = y_axis;
%         else
%             xx = cat(1, xx, x_axis);
%             yy = cat(1, yy, y_axis);
%         end
    end
     for kk=1:LL
%         set (gcf, 'position', [0, 0, 0, 0]);
        cell = floor(c{kk,1} * 1);
        x_axis = cell(:, 1);
        y_axis = cell(:, 2);
        x = find(x_axis < 1);
        y = find(y_axis < 1);
        if ~isempty(x)
            for nn = 1: length(x)
                x_axis(x(nn)) = 1;
            end
        end
        if ~isempty(y)
            for nn = 1: length(y)
                y_axis(y(nn)) = 1;
            end
        end
        x = find(x_axis > 1000);
        y = find(y_axis > 1000);
        if ~isempty(x)
            for nn = 1: length(x) 
                x_axis(x(nn)) = 1000;
            end
        end
        if ~isempty(y)
            for nn = 1: length(y)
                y_axis(y(nn)) = 1000;
            end
        end
        for kkk = 1:length(cell)
            mask_g(x_axis(kkk), y_axis(kkk)) = 255;
        end
%         axis(gca,'off');
%         plot(x_axis, y_axis);
%         hold on
%         if kk = 1
%             xx = x_axis;
%             yy = y_axis;
%         else
%             xx = cat(1, xx, x_axis);
%             yy = cat(1, yy, y_axis);
%         end
    end
% set(gcf, 'unit', 'pixels', 'position', [0 0 2000 2000]);
% set(gca, 'Position', [0 0 2000 2000]);
mask(:, :, 2) = mask_g;
imwrite(mask, [out_Path, oneXml(44:end-4), '.tif']);
end