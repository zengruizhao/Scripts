function [features sum] = cellxy(cell,image,windsize,featchoice)
% ���ڴ�����ȡ����
%featursΪ����Ҫ��ȡ��������sumΪ����������windsize���롾Ϊ������
windin=floor(windsize/2); %С
windout=ceil(windsize/2); %��
[zb l n]=bwboundaries(cell);% �ⲿ����
features=[];
[m l z]=size(image);
sum=0;
for i=1:n
 z=zb(i,1);
 z=cell2mat(z);
 x1=z(:,2);  %������ͨ��������
 y1=z(:,1); 
 xy=[x1 y1];
 [cx cy A]=centroid(xy);  %cell��������
 if((windout<cx&&cx<(l-windout))&&(windout<cy&&cy<(m-windout)))  %ȥ��ͼ��߽總����ϸ��
   cell=image(cy-windin:cy+windin,cx-windin:cx+windin,:);  %ѡȡ���ڴ�С��ϸ������  
   feature=choicefeat(cell,featchoice);
   features =[features; feature];   
   sum=sum+1; %ͳ����������
 end 
end

