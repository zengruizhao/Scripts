function [location] = cellocationl(cell)
% 获取样本细胞坐标
se=[0 1 0;1 1 1;0 1 0];
label=double(cell);
R_l=label(:,:,1);
R_l=imdilate(R_l,se);%
R_l=imdilate(R_l,se);
[zb l n]=bwboundaries(R_l);% 外部轮廓
for i=1:n
    z=zb(i,1);
    z=cell2mat(z);
    xy=[z(:,2) z(:,1)];
    [cx cy A]=centroid(xy);  %cell中心坐标
    location(i,:) = [cx,cy];
end
