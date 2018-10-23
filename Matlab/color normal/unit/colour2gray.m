function image=colour2gray(scoreImage)
 %ÅÐ¶ÏÊÇ·ñÎª²ÊÉ«Í¼Ïñ£¬ÈôÊÇ²ÊÉ«Í¼ÏñÔò½«²ÊÉ«Í¼Ïñ×ª»»Îª»Ò¶ÈÍ¼Ïñ
coloursize=size(scoreImage);
if numel(coloursize)>2
  image=rgb2gray(scoreImage); 
else
  image=scoreImage;
end