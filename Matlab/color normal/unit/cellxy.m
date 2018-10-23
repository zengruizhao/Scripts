function [features sum] = cellxy(cell,image,windsize,featchoice)
% 用于窗口提取特征
%featurs为所需要提取的特征，sum为样本数量，windsize必须【为奇数】
windin=floor(windsize/2); %小
windout=ceil(windsize/2); %大
[zb l n]=bwboundaries(cell);% 外部轮廓
features=[];
[m l z]=size(image);
sum=0;
for i=1:n
 z=zb(i,1);
 z=cell2mat(z);
 x1=z(:,2);  %单个联通区域坐标
 y1=z(:,1); 
 xy=[x1 y1];
 [cx cy A]=centroid(xy);  %cell中心坐标
 if((windout<cx&&cx<(l-windout))&&(windout<cy&&cy<(m-windout)))  %去除图像边界附近的细胞
   cell=image(cy-windin:cy+windin,cx-windin:cx+windin,:);  %选取窗口大小的细胞区域  
   feature=choicefeat(cell,featchoice);
   features =[features; feature];   
   sum=sum+1; %统计区块数量
 end 
end

