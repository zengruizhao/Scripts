function [t]=statxture(f,scale) 
 
%STATXTURE Computes statistical measures of texture in an image.
%   T = STATXURE(F, SCALE) computes six measures of texture from an
%   image (region) F. Parameter SCALE is a 6-dim row vector whose 
%   elements multiply the 6 corresponding elements of T for scaling
%   purposes. If SCALE is not provided it defaults to all 1s.  The 
%   output T is 6-by-1 vector with the following elements: 
%     T(1) = Average gray level
%     T(2) = Average contrast
%     T(3) = Measure of smoothness
%     T(4) = Third moment
%     T(5) = Measure of uniformity
%     T(6) = Entropy
%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.5 $  $Date: 2004/11/04 22:33:43 $
if nargin==1 
    scale(1:6)=1; 
else 
    scale=scale(:)';  % Make sure it's a row vector
end 
%Obtain histogram and normalize it.
f=colour2gray(f);
p=imhist(f);                  %p是256*1的列向量 
p=p./numel(f); 
L=length(p); 
% Compute the three moments. We need the unnormalized ones
% from function statemoments. There are in vector mu.
[v,mu]=statmoments(p,3); 
%计算六个纹理特征  Compute the six texture measures
t(1)=mu(1);                   %平均值  Average gray level
t(2)=mu(2).^0.5;              %标准差   Standard deviation
varn=mu(2)/(L-1)^2;           % First normalize the variance to [0 1] by dividing it by (L-1)^2.
t(3)=1-1/(1+varn);            %平滑度首先为（0~1）区间通过除以（L-1）^2将变量标准化 
t(4)=mu(3)/(L-1)^2;           %三阶矩（通过除以（L-1）^2将变量标准化） Third moment (normalized by (L-1)^2 also) 
t(5)=sum(p.^2);               %一致性  Uniformity
t(6)=-sum(p.*(log2(p+eps)));  %熵  Entropy
T=[t(1) t(2) t(3) t(4) t(5) t(6)];
%缩放值，默认为1  Scale the value
t=t.*scale; 
end 
 
