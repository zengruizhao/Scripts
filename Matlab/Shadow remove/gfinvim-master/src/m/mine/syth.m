clear;clc;
load H 
load S
%%
load Iout
V = double(Iout)./255;
img(:,:,1) = H;
img(:,:,2) = S;
img(:,:,3) = V;
out = hsv2rgb(img);
imshow(out);

%%
% load V
% img(:,:,1) = H;
% img(:,:,2) = S;
% img(:,:,3) = V;
% out = hsv2rgb(img);
% imshow(out);