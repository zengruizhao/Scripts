clear;clc;
img = imread('ISIC_0000000_superpixels.png');
index = decodeSuperpixelIndex(img);
% index = uint32(index);
[col_val,ind_first]=unique(index,'first');