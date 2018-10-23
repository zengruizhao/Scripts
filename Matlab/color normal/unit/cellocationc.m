function [location] = cellocationc(cell)
% ��ȡ����ϸ������
se=[0 1 0;1 1 1;0 1 0];
label=double(cell);
G_l=label(:,:,2);
G_l=imdilate(G_l,se);
G_l=imdilate(G_l,se);
[zb l n]=bwboundaries(G_l);% �ⲿ����
for i=1:n
    z=zb(i,1);
    z=cell2mat(z);
    xy=[z(:,2) z(:,1)];
    [cx cy A]=centroid(xy);  %cell��������
    location(i,:) = [cx,cy];
end

