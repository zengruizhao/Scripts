function [location] = cellocationc(cell)
% 获取样本细胞坐标
se=[0 1 0;1 1 1;0 1 0];
label=double(cell);
G_l=label(:,:,2);
G_l=imdilate(G_l,se);
G_l=imdilate(G_l,se);
[zb l n]=bwboundaries(G_l);% 外部轮廓
for i=1:n
    z=zb(i,1);
    z=cell2mat(z);
    xy=[z(:,2) z(:,1)];
    [cx cy A]=centroid(xy);  %cell中心坐标
    location(i,:) = [cx,cy];
end

