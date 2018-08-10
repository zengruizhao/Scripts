clear;clc
img = imread('temp.jpg');
%%
% Haematoxylin and Eosin (H&E) x2
% Haematoxylin and DAB (H DAB)
% Feulgen Light Green
% Giemsa
% Fast Red, Fast Blue and DAB
% Methyl green and DAB
% Haematoxylin, Eosin and DAB (H&E DAB)
% Haematoxylin and AEC (H AEC)
% Azan-Mallory
% Masson Trichrome
% Alcian blue & Haematoxylin
% Haematoxylin and Periodic Acid - Schiff (PAS)
% RGB subtractive
% CMY subtractive
%%
[ DCh, M ] = Deconv(img, 'HE');
[h, w, c] = size(DCh);
Channel = DCh(:,:,1);
Channel = (Channel-min(Channel(:))) / (max(Channel(:)) - min(Channel(:))) ;
bw = im2bw(Channel(:,:,1), graythresh(Channel(:,:,1)));
%%
r=ones(1001, 1);
g =  1:-0.001:0;
b =  1:-0.001:0;
colormap = [r, g', b'];
imshow(Channel, 'Colormap', colormap);
 