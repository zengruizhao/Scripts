function stats = nFold_FS(data_set,data_labels,shuffle,n,nIter,Subsets,nftrs,FSmethod)
% Using n-fold feature selection 
% Input:
%   data_set: data
%   data_labels: labels
%   shuffle: 1 for random, 0 for non-random partition (Default: 1)
%   n: Number of folds to your cross-validation (Default: 3)
%   nIter: Number of cross-validation iterations (Default: 1)
%   Subsets: pass your own training and testing subsets & labels (Default:
%   computer will generate using 'nFold')
%   nftrs: number of features to select
%   FSmethod: string indicating feature selection algorithm to use
%       'mrmr' -> uses "mrmr_mid_d.m" from the repository
%     Can also use any FS algorithm in MATLAB's "rankfeatures" function,
%     from its documentation: 
%       'ttest' (default) ?Absolute value two-sample t-test with pooled variance estimate.
%       'entropy' ?Relative entropy, also known as Kullback-Leibler distance or divergence.
%       'bhattacharyya' ?Minimum attainable classification error or Chernoff bound.
%       'roc' ?Area between the empirical receiver operating characteristic (ROC) curve and the random classifier slope.
%       'wilcoxon' ?Absolute value of the standardized u-statistic of a two-sample unpaired Wilcoxon test, also known as Mann-Whitney.
%    *'ttest', 'entropy', and 'bhattacharyya' assume normal distributed classes while 'roc' and 'wilcoxon' are nonparametric tests. All tests are feature independent.
% Output:
%   stats: struct containing features selected for each fold for each iteration.
% Nate Braman, 2017

% data_set = eigVec_graphfeats;
% data_labels = labels(:);
% classifier = 'SVM';
% shuffle = 1;
% n = 3;
if nargin < 6
    Subsets = {};
end
if nargin < 5
    nIter = 1;
end
if nargin < 4
    n = 3; % 3-fold cross-validation
end
if nargin < 3
    shuffle = 1; % randomized
end

% if any(~xor(data_labels == 1, data_labels == -1)), error('Labels must be 1 and -1'); end

stats = struct; %cell(1,nIter);
for j=1:nIter
    fprintf('Iteration: %i\n',j);
    
    % reset total statistics
    Ttp = 0; Ttn = 0; Tfp = 0; Tfn = 0; decision=zeros(size(data_labels)); prediction=zeros(size(data_labels));
    
    if isempty(Subsets)
        [tra tes]=GenerateSubsets('nFold',data_set,data_labels,shuffle,n);
    else
        tra = Subsets{j}.training;
        tes = Subsets{j}.testing;
    end
    stats(j).ftrs = [];
    stats(j).AUC_FF = [];
    stats(j).AUC = [];
    for i=1:n
        %fprintf(['Fold # ' num2str(i) '\n']);
        fprintf('Iteration: %i\n',j);
        training_set = data_set(tra{i},:);
        testing_set = data_set(tes{i},:);
        training_labels = data_labels(tra{i});
        testing_labels = data_labels(tes{i});
        
        if strcmp(FSmethod, 'mrmr')
            fea = mrmr_mid_d(training_set,training_labels,nftrs); % MRMR
            fea = fea'; 

        else
            [IDX, Z] = rankfeatures(training_set',training_labels,'criterion',FSmethod,'CCWeighting', 1); %%CCWeighting is the parameter that weights by correlation w/ previously selected features
            fea = IDX(1:nftrs);
        end
        stats(j).ftrs = [stats(j).ftrs, fea'];
    end

end

end
