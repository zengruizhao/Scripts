%从RGB空间到HSV空间转换,以及查看H,S,V分量的Matlab程序实现
clear;clc;
a=imread('112.png'); 
hv=rgb2hsv(a); 
%可以通过下面的程序看一幅图的HSV三个通道 
H=hv(:,:,1);
S=hv(:,:,2);
V=hv(:,:,3);

figure;
subplot(1,2,1);imshow(a);title('原始图像'); 
subplot(1,2,2);imshow(hv);title('HSV空间图像');

figure;
subplot(1,3,1);imshow(H);title('HSV空间H分量图像');
subplot(1,3,2);imshow(S);title('HSV空间S分量图像');
subplot(1,3,3);imshow(V);title('HSV空间V分量图像');
imwrite(V,'v.png');