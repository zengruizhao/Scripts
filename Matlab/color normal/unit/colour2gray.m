function image=colour2gray(scoreImage)
 %�ж��Ƿ�Ϊ��ɫͼ�����ǲ�ɫͼ���򽫲�ɫͼ��ת��Ϊ�Ҷ�ͼ��
coloursize=size(scoreImage);
if numel(coloursize)>2
  image=rgb2gray(scoreImage); 
else
  image=scoreImage;
end