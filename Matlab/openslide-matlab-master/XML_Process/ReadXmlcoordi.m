clear all;
clc;
close all;

addpath('/home/xjw/Nuclear_Seg_SegNet/Code/XML_Process/');
xmlPath='/home/xjw/Nuclear_Seg_SegNet/Data/train/xml/';
out_Path = '/home/xjw/Nuclear_Seg_SegNet/Data/train/mask/';
ImgSize = 1000;

xml = dir([xmlPath, '*.xml']); 
for i = 6:length(xml)
    xmlName = xml(i).name;
    xmlFile = [xmlPath, xmlName];
    mask_g = uint8(zeros(1000, 1000));
     X_boundary = [];
     Y_boundary = [];
    c = parseCamelyon16XmlAnnotations_ccf(xmlFile);
    for j = 1:length(c);
        cell = c{j, 1};
        x = cell(:, 1);
        y = cell(:, 2);
        TempImg = poly2mask(x, y, ImgSize, ImgSize);
        B = bwboundaries(TempImg);
        if ~isempty(B)
            B = B{1, 1};
            X_boundary = cat(1, X_boundary, B(:, 1));
            Y_boundary = cat(1, Y_boundary, B(:, 2));
        end
        TempImg = uint8(TempImg);
        [XTemp, YTemp] = find(TempImg == 1);       
        for k = 1:length(XTemp)
            mask_g(XTemp(k), YTemp(k)) = 255;
        end
    end
    for l = 1:length(X_boundary)
        mask_g(X_boundary(l), Y_boundary(l)) = 128;
    end    
    imwrite(mask_g, [out_Path, xmlName(1:end-4), '.tif']);
end