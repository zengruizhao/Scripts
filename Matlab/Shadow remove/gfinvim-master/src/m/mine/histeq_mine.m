clear;clc;
% load T;
img = imread('instrinsic.png');
[I,T]=histeq(img);
% b = grayxformmex(img,T);
imshow(I);
