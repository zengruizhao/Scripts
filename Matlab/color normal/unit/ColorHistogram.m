%**************************************************************************
%                 图像检索——提取颜色特征
%HSV空间颜色直方图(将RGB空间转化为HSV空间并进行非等间隔量化，
%将三个颜色分量表示成一维矢量，再计算其直方图作为颜色特征
%function : Hist = ColorHistogram(Image)
%Image    : 输入图像数据
%Hist     : 返回颜色直方图特征向量36维
%**************************************************************************
function Hist = ColorHistogram(Image)
%Image=imread('toy.bmp');
%imshow('plane.bmp');
%Image=imread('panda.bmp');
%imshow('rgb.tif');
[M,N,O]=size(Image);
[h,s,v]=rgb2hsv(Image);
H = h; S = s; V = v;
h=h*360;
%when v<0.2,it is a black area;when s<0.2&0.2=<v<0.8,it is a gray area
for i=1:M
    for j=1:N
        if v(i,j)<0.2
           L(i,j)=0;
         end
        if s(i,j)<0.2&&v(i,j)>0.2&&v(i,j)<=0.8
            L(i,j)=(v(i,j)-0.2)*10+1;
        end
        if s(i,j)<0.2&&v(i,j)>0.8&&v(i,j)<=1 
                    L(i,j)=7; %white area
        end
     end
end
%*************************************************

%将hsv空间非等间隔量化：color area
%  h量化成7级；Similar to the vision model
%  s量化成2级；
%  v量化成2级；
for i = 1:M
    for j = 1:N
        if h(i,j)>330&&h(i,j)<=360||h(i,j)<=22
                H(i,j) = 0;
         end
        if h(i,j)>22&&h(i,j)<=45
            H(i,j)=1;
        end
        if h(i,j)>45&&h(i,j)<=70
            H(i,j)=2;
        end 
         if h(i,j)>70&&h(i,j)<=155
            H(i,j)=3;
        end
         if h(i,j)>155&&h(i,j)<=186
            H(i,j)=4;
        end
         if h(i,j)>186&&h(i,j)<=278
            H(i,j)=5;
        end
         if h(i,j)>278&&h(i,j)<=330
            H(i,j)=6;
        end
    end
end
for i = 1:M
      for j = 1:N  
            if s(i,j)>0.2&&s(i,j)<=0.65
                S(i,j)=0;
            end
            if s(i,j)>0.65&&s(i,j)<=1
                S(i,j)=1;
          end
   end
end
  for i=1:M
      for j=1:N
          if v(i,j)>0.2&&v(i,j)<=0.7
              V(i,j)=0;
          end
          if v(i,j)>0.7&&v(i,j)<=1
              V(i,j)=1;
          end
      end
  end
%将三个颜色分量合成为一维特征向量：L=4*H+2*S+V+8
for i=1:M
     for j=1:N
         if s(i,j)>0.2&&s(i,j)<=1&&v(i,j)>0.2&&v(i,j)<=1
             L(i,j)=4*H(i,j)+2*S(i,j)+V(i,j)+8;
         end
     end
end
%计算L的直方图
for i=0:35
    Hist(i+1)=size(find(L==i),1);
end
Hist = Hist/sum(Hist);
