% 自设计特征
function feature=featbylei(image)
%将 图像缩小1/3

im=imresize(image,[7 7]);
imblue=blueratio(im);
% ave=mean(imblue(:));
% variance=var(imblue(:));
hh=size(find(imblue>(1.25*mean(imblue(:)))),1); %全局与均值的比较
ll=size(find(imblue<(0.5*mean(imblue(:)))),1); %全局与均值的比较
dif1=mean(mean(imblue(3:5,3:5)))-mean(mean(imblue([1 7 43 49 4 22 28 46]))); %中心与边界的比较
dif2=mean(mean(imblue([18 24 26 32])))-mean(mean(imblue([9 13 37 41]))); %内层的比较
dif3=mean(mean(imblue([17 19 31 33])))-mean(mean(imblue([11 23 27 39]))); %内层的比较
% mid=mean(mean(imblue(3:5,3:5)));
edge=imblue([1 2 3 4 5 6 7 8 14 15 21 22 28 29 35 36 42 43 44 45 46 47 48 49]);
nu1=size(find(edge>mean(mean(imblue(3:5,3:5)))),2);  %%中心与边界的比较
nu2=size(find(edge>mean(imblue(:))),2); %均值与边界的比较
t=1;
if size(find(imblue>mean(mean(imblue(3:5,3:5)))),1)>10;  %细胞区域的大小
    t=0;
end
% m=1;
% if size(find(imblue>mean(mean(imblue(3:5,3:5)))),1)<6; %细胞区域的大小
%     m=0;
% end
[x y]=find(imblue==max(max(imblue)));
% num=6*(y(1)-1)+x(1);
% dis=abs(4-x(1))+abs(4-y(1));
dis=((4-x(1))^2+(4-y(1))^2)^0.5;
% feature=[hh ll dif1 dif2 mid nu1 nu2 t m num];
feature=[hh ll dif1 dif2 dif3 nu1 nu2 t dis];

%best 1.5 0.5 3.65