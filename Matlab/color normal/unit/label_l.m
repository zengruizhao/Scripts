function xy=label_l(labelim)
%提取label图像标记点的信息
label=double(labelim);
R_l=label(:,:,1);
G_l=label(:,:,2);
B_l=label(:,:,3);
xy=R_l;%l
% xy=G_l;%c