function feature=choicefeat(image,featchoice)
switch featchoice
  case 1
    txtu = statxture(image) ;  %纹理特征 1*6维
    gs = gsjz(image);  %灰度共生矩阵  1*8维
    feature=[txtu gs]; 
  case 2
    feature=statxture(image) ; %纹理
  case 3
%     feature1 = colorMoments(image);  %颜色矩
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
