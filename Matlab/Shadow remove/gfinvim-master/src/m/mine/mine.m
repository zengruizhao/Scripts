%��RGB�ռ䵽HSV�ռ�ת��,�Լ��鿴H,S,V������Matlab����ʵ��
clear;clc;
a=imread('112.png'); 
hv=rgb2hsv(a); 
%����ͨ������ĳ���һ��ͼ��HSV����ͨ�� 
H=hv(:,:,1);
S=hv(:,:,2);
V=hv(:,:,3);

figure;
subplot(1,2,1);imshow(a);title('ԭʼͼ��'); 
subplot(1,2,2);imshow(hv);title('HSV�ռ�ͼ��');

figure;
subplot(1,3,1);imshow(H);title('HSV�ռ�H����ͼ��');
subplot(1,3,2);imshow(S);title('HSV�ռ�S����ͼ��');
subplot(1,3,3);imshow(V);title('HSV�ռ�V����ͼ��');
imwrite(V,'v.png');