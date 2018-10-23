function labelfeature=labeljz(n_l,n_c)
% 对应于样本集特征矩阵的label信息矩阵，训练用 暂时无用
labelfeature_l=ones(n_l,1);
labelfeature_c=-ones(n_c,1);
labelfeature=[labelfeature_l; labelfeature_c];