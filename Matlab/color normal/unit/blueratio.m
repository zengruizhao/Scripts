function [Br Rr Gr]=blueratio(I)
I=double(I);
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);
Br=((200.*B)./(1+R+G)).*(256./(1+B+R+G));
Rr=((200.*R)./(1+B+G)).*(256./(1+B+R+G));
Gr=((200.*G)./(1+R+B)).*(256./(1+B+R+G));