clear;clc;
path = '/media/zzr/SW/Skin_xml/Patch/2018-08-09 153304/';
img_all = dir([path '*.jpg']);
color = [0, 0, 0;
    255, 0, 0];
for i=1:length(img_all)
   img = imread([path img_all(i).name]);
%    mask = imread([path 'mask/' img_all(i).name]);
    mask = imread([path img_all(i).name(1:end-4) '_nuSeg1.png']);
    mask = rgb2gray(mask);
    threshold = graythresh(mask);
    % threshold = adaptthresh(ChannelHistEqual, 'NeighborhoodSize', 41);
    bw = imbinarize(mask, threshold);
    bw = bwareaopen(bw, 100);
    bw = imfill(bw, 'holes');
    coord = bwboundaries(bw);
    imshow(img,'InitialMagnification',32);hold on;
    for j=1:length(coord)
       boundary = coord{j};
       plot(boundary(:, 2), boundary(:, 1), 'g', 'LineWidth', 1); 
    end
    pause
%    r = zeros(size(mask));
%    g = zeros(size(mask));
%    b = zeros(size(mask));
%    for j=1:2 % 3
%        r(logical(mask==j-1)) = color(j, 1);
%        g(logical(mask==j-1)) = color(j, 2);
%        b(logical(mask==j-1)) = color(j, 3);
%    end
%    mask = cat(3, r, g, b);
%    imshow(img);hold on 
%    imshow(mask, []); alpha(0.65);hold off
%    pause;
% subplot(121);imshow(img);
% subplot(122);imshow(mask, []);
% pause;
end
%%
% clear;clc;
% path = '/media/zzr/Data/skin_xml/';
% img_all = dir([path 'semantic/*.png']);
% color = [0, 0, 0;
%     255, 0, 0];
% for i=1:length(img_all)
%    img = imread([path 'semantic/' img_all(i).name]);
%    mask = im2double(imread([path 'semantic_result/' img_all(i).name(1:end-4) '_seg.png']));
%    r = zeros(size(mask));
%    g = zeros(size(mask));
%    b = zeros(size(mask));
%    for j=1:2
%        r(logical(mask==j-1)) = color(j, 1);
%        g(logical(mask==j-1)) = color(j, 2);
%        b(logical(mask==j-1)) = color(j, 3);
%    end
%    mask = cat(3, r, g, b);
%    imshow(img);hold on 
%    imshow(mask, []); alpha(0.65);hold off
%    pause;
% end