function feature=choicefeat(image,featchoice)
switch featchoice
  case 1
    txtu = statxture(image) ;  %�������� 1*6ά
    gs = gsjz(image);  %�Ҷȹ�������  1*8ά
    feature=[txtu gs]; 
  case 2
    feature=statxture(image) ; %����
  case 3
%     feature1 = colorMoments(image);  %��ɫ��
%     feature2=featbylei(image);
%     feature=[feature1 feature2];

    feature=featbylei(image);

  case 4
    gray=rgb2gray(image);
    MAPPING=getmapping(16,'riu2');
    feature=lbp(gray,2,16,MAPPING,'hist');%LBP
  case 5
    feature = ImgHOGFeature(image)'; %hog
%     feature=image(:)';
end
