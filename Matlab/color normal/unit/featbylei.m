% ���������
function feature=featbylei(image)
%�� ͼ����С1/3

im=imresize(image,[7 7]);
imblue=blueratio(im);
% ave=mean(imblue(:));
% variance=var(imblue(:));
hh=size(find(imblue>(1.25*mean(imblue(:)))),1); %ȫ�����ֵ�ıȽ�
ll=size(find(imblue<(0.5*mean(imblue(:)))),1); %ȫ�����ֵ�ıȽ�
dif1=mean(mean(imblue(3:5,3:5)))-mean(mean(imblue([1 7 43 49 4 22 28 46]))); %������߽�ıȽ�
dif2=mean(mean(imblue([18 24 26 32])))-mean(mean(imblue([9 13 37 41]))); %�ڲ�ıȽ�
dif3=mean(mean(imblue([17 19 31 33])))-mean(mean(imblue([11 23 27 39]))); %�ڲ�ıȽ�
% mid=mean(mean(imblue(3:5,3:5)));
edge=imblue([1 2 3 4 5 6 7 8 14 15 21 22 28 29 35 36 42 43 44 45 46 47 48 49]);
nu1=size(find(edge>mean(mean(imblue(3:5,3:5)))),2);  %%������߽�ıȽ�
nu2=size(find(edge>mean(imblue(:))),2); %��ֵ��߽�ıȽ�
t=1;
if size(find(imblue>mean(mean(imblue(3:5,3:5)))),1)>10;  %ϸ������Ĵ�С
    t=0;
end
% m=1;
% if size(find(imblue>mean(mean(imblue(3:5,3:5)))),1)<6; %ϸ������Ĵ�С
%     m=0;
% end
[x y]=find(imblue==max(max(imblue)));
% num=6*(y(1)-1)+x(1);
% dis=abs(4-x(1))+abs(4-y(1));
dis=((4-x(1))^2+(4-y(1))^2)^0.5;
% feature=[hh ll dif1 dif2 mid nu1 nu2 t m num];
feature=[hh ll dif1 dif2 dif3 nu1 nu2 t dis];

%best 1.5 0.5 3.65