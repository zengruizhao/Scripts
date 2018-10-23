function [trainfeat_l trainfeat_c]=trainmatrix(imageori , label_ll, label_cc, windsize,featchoice)
%读入原始图像，两类细胞的标记图像生成样本特征矩阵

% 淋巴细胞label图像，获取坐标点
xy_l=label_l(label_ll);  %标记点坐标矩阵

[feat_l n_l] = cellxy(xy_l,imageori,windsize,featchoice); %正样本特征矩阵，n为样本数量

% 上皮细胞label图像，获取坐标点
xy_c=label_c(label_cc);  %标记点坐标矩阵truth_masktruth_masktruth_masktruth_masktruth_masktruth_mask

[feat_c n_c] = cellxy(xy_c,imageori,windsize,featchoice); %负样本特征矩阵，n为样本数量

% 各类特征矩阵
trainfeat_l=feat_l;
trainfeat_c=feat_c;
