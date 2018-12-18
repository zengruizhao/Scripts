function [h, p, ci] = PairedTTest(data1, labels1, data2, labels2, nFolds, kernel, nIter)

% load data & labels
% load /data2/home/abasavan/for_scott/LI_autoseg_graphfeats_3D;
% load /data2/home/abasavan/for_scott/LI_manseg_graphfeats_3D;

% check inputs
M = size(data1,1);
if size(data2,1) ~= M || size(labels1,1) ~= M || size(labels2,1) ~= M
    error('Number of samples do not match!');
end

% run cross-validation on autoseg
autoseg_stats = CrossValidate('nFold_SVM',data1,labels1,1,nFolds,kernel,nIter);
[meanacc stdacc] = getMeanStdAccFromStats(autoseg_stats);

% save subsets that were randomly generated in cross-validation
Subsets = cell(1,length(autoseg_stats));
for i=1:length(autoseg_stats)
    Subsets{i} = autoseg_stats{i}.subsets;
end


% run cross-validation on manseg using subsets from autoseg
manseg_stats = CrossValidate('nFold_SVM',data2,labels2,1,nFolds,kernel,nIter,Subsets);
[meanacc stdacc] = getMeanStdAccFromStats(manseg_stats);

% paired ttest
a=[]; b=[];
for i=1:length(autoseg_stats)
    a = [a autoseg_stats{i}.acc];
    b = [b manseg_stats{i}.acc];
end

[h,p,ci] = ttest2(a,b,.05,'both','unequal');