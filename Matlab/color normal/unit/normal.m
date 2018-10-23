function images=normal(image)
%用于blueratio标准化
image=(image-min(min(image)))/(max(max(image))-min(min(image)))*255;
image=floor(image);
images=double(image);