function [location] = cellocationl(cell)
% ��ȡ����ϸ������
se=[0 1 0;1 1 1;0 1 0];
label=double(cell);
R_l=label(:,:,1);
R_l=imdilate(R_l,se);%
R_l=imdilate(R_l,se);
[zb l n]=bwboundaries(R_l);% �ⲿ����
for i=1:n
    z=zb(i,1);
    z=cell2mat(z);
    xy=[z(:,2) z(:,1)];
    [cx cy A]=centroid(xy);  %cell��������
    location(i,:) = [cx,cy];
end
