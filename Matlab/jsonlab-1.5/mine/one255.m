B=zeros(767,1022);
for i=1:767
    for j=1:1022
        if ISIC_0000000(i,j)~=0
            B(i,j)=255;
        end
    end
end
imshow(B);