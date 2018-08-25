function cmVec = colorMom_zzr(jpgfile)  
% function cmVec = colorMom(direc)  
% Input: 3d image   
% Output: a color moment feature vector of the input image.
% Function RGB2Lab (By Mr. Ruzon in Stanford U.) is used in this function.  
   
[m,n,~] = size(jpgfile);   
cmVec = zeros(1,9);   
cmVec(1) = mean(mean(jpgfile(:,:,1)));  
cmVec(2) = mean(mean(jpgfile(:,:,2)));  
cmVec(3) = mean(mean(jpgfile(:,:,3)));  
% cal Moment 2 and 3  
for p = 1:m  
   for q = 1:n  
       % === Moment2  
       cmVec(4) = cmVec(4) + (jpgfile(p,q,1)-cmVec(1))^2;  
       cmVec(5) = cmVec(5) + (jpgfile(p,q,2)-cmVec(2))^2;  
       cmVec(6) = cmVec(6) + (jpgfile(p,q,3)-cmVec(3))^2;  
       % === Moment3  
       cmVec(7) = cmVec(7) + (jpgfile(p,q,1)-cmVec(1))^3;  
       cmVec(8) = cmVec(8) + (jpgfile(p,q,2)-cmVec(2))^3;  
       cmVec(9) = cmVec(9) + (jpgfile(p,q,3)-cmVec(3))^3;                  
   end  
end  
cmVec(4:9) = cmVec(4:9)/(m*n);  
cmVec(4) = cmVec(4)^(1/2);  
cmVec(5) = cmVec(5)^(1/2);  
cmVec(6) = cmVec(6)^(1/2);  
  
if cmVec(7) >0  
   cmVec(7) = cmVec(7)^(1/3);  
else  
   cmVec(7) = -((-cmVec(7))^(1/3));  
end  
if cmVec(8) >0  
   cmVec(8) = cmVec(8)^(1/3);  
else  
   cmVec(8) = -((-cmVec(8))^(1/3));  
end  
if cmVec(9) >0  
   cmVec(9) = cmVec(9)^(1/3);  
else  
   cmVec(9) = -((-cmVec(9))^(1/3));  
end   
        
% Normalize...  
if sqrt(sum(cmVec.^2))~=0  
    cmVec = cmVec / sqrt(sum(cmVec.^2));  
end  
