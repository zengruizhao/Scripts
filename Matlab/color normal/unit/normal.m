function images=normal(image)
%����blueratio��׼��
image=(image-min(min(image)))/(max(max(image))-min(min(image)))*255;
image=floor(image);
images=double(image);