function [trainfeat_l trainfeat_c]=trainmatrix(imageori , label_ll, label_cc, windsize,featchoice)
%����ԭʼͼ������ϸ���ı��ͼ������������������

% �ܰ�ϸ��labelͼ�񣬻�ȡ�����
xy_l=label_l(label_ll);  %��ǵ��������

[feat_l n_l] = cellxy(xy_l,imageori,windsize,featchoice); %��������������nΪ��������

% ��Ƥϸ��labelͼ�񣬻�ȡ�����
xy_c=label_c(label_cc);  %��ǵ��������truth_masktruth_masktruth_masktruth_masktruth_masktruth_mask

[feat_c n_c] = cellxy(xy_c,imageori,windsize,featchoice); %��������������nΪ��������

% ������������
trainfeat_l=feat_l;
trainfeat_c=feat_c;
