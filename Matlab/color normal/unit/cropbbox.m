function imgout=cropbbox(img,bbox)

% imgout=cropbbox(img,bbox)
% 
% crop posibbly multi-chanel image by a
% bounding box bbox=[xmin,ymin,xmax,ymax].
% If the bounding box exceeds image dimensions
% the image is replicated by mirroring image
% pixels outside image area.

bbox=round(bbox);
[ysz,xsz,csz]=size(img);

if bbox(1)>=1 & bbox(3)<=xsz & ...
   bbox(2)>=1 & bbox(4)<=ysz

  imgout=img(bbox(2):bbox(4),bbox(1):bbox(3),:);
else
  
  % fill unknown pixels with the mirrored image
  [x,y]=meshgrid(bbox(1):bbox(3),bbox(2):bbox(4));
  go=1;
  while go
    go=0;
    ix=find(x(:)<1); iy=find(y(:)<1);
    if length(ix) x(ix)=abs(x(ix))+2; end
    if length(iy) y(iy)=abs(y(iy))+2; end
    ix=find(x(:)>xsz); iy=find(y(:)>ysz); 
    if length(ix) x(ix)=2*xsz-x(ix)-1; go=1; end
    if length(iy) y(iy)=2*ysz-y(iy)-1; go=1; end
  end
  % handle n-dimensional images
  szorig=size(img);
  img=reshape(img,xsz*ysz,csz);
  ind=sub2ind([ysz,xsz],y(:),x(:));
  sz=size(x);
  if length(szorig>2) sz=[sz szorig(3:end)]; end
  imgout=reshape(img(ind,:),sz);
end