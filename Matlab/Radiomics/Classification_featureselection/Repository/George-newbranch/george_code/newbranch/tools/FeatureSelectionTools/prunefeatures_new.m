function [datasubset,confidence] = prunefeatures_new(data, labels_cheng, type, threshold, sig)
% wrapper function for feature selection
% outputs:
% datasubset = feature index ranked in order of importance
% confidence = corresponding scores given by feature selection type
%
% inputs:
% data = n x d matrix (samples x dimensionality)
% labels_cheng = n x 1 matrix (binary class, numeric only)
% type = support for ttest, wilcoxon rank sum, f-test, wilks-lambda, and
% Kolmogorov?Smirnov test
% threshold = can be set as number of features kept (threshold > 1) or as a
% percentage (threshold < 1)
% sig = alternative thresholding based on a minimum score requirement
% (sig = 0.05 keeps features with p-values < 0.05)
%
% George Lee (2014)

if nargin < 3
    type = 'ttestp'; %default
end

if nargin < 4
    threshold = size(data,2); % default
end

if threshold < 1 % can also treat threshold input as a percentage
    threshold = ceil(size(data,2)*threshold);
end

% test unique
if length(unique(labels_cheng)) > 2
    fprintf('More than 2 types of labels_cheng found! Comparing significance of first two found');
elseif length(unique(labels_cheng)) < 2
    fprintf('Not enough label types! Need two different labels_cheng');
else
    [lab] = unique(labels_cheng);
end

if ~strcmp(type,'mrmr')
    for i = 1:size(data,2)
%         i=228
        [score] = fswrapper(data(:,i), labels_cheng, type);
        confidence(i) = score;
    end
else
    dataw_discrete=makeDataDiscrete_mrmr(data);
    datasubset= mrmr_mid_d(dataw_discrete, labels_cheng, 50);
%     datasubset=dataw_discrete(:,index);
    confidence=50:-1:1;
    return;
end

% features(1:length(find(confidence <= sig))) = find(confidence <= sig);
% pvalues(1:length(find(confidence <= sig))) = confidence(confidence <= sig);

%threshold = 25;
switch type
    case 'ftest'
        [confidence,index] = sort(confidence);
    case 'ttestp'
        [confidence,index] = sort(confidence);
    case {'wilcoxon' ,'wilkcox'}
        [confidence,index] = sort(confidence);
    case 'mrmr'
        [confidence,index] = sort(confidence);
    otherwise
        [confidence,index] = sort(confidence,'descend');
end

%remove nans and append to end for reference
        bad = find(isnan(confidence));
        index(end+1:end+numel(bad))= index(bad);
        confidence(end+1:end+numel(bad))= confidence(bad);
        index(bad)= [];
        confidence(bad)= [];
        
if nargin < 5
    if threshold > numel(index)
        features = index;
    else
        features = index(1:threshold);
    end
else
    %sig = 0.05 % default
    features = index(confidence < sig);
end


datasubset = features;


function [score,index] = fswrapper(data, unprocessedlabels_cheng, type)

[lab] = unique(unprocessedlabels_cheng);
if isnumeric(lab)||islogical(lab)
    labels_cheng(unprocessedlabels_cheng == lab(1)) = lab(1);
    labels_cheng(unprocessedlabels_cheng == lab(2)) = lab(2);
elseif iscell(lab)
    labels_cheng(ismember(unprocessedlabels_cheng,lab{1})) = 1;
    labels_cheng(ismember(unprocessedlabels_cheng,lab{2})) = 2;
end

switch type
    case 'ttestp' % student t
        [h,score] = ttest2(data(labels_cheng==lab(1),:),data(labels_cheng==lab(2),:));
    case 'ttestt'
        [score,index] = ttestnew(data, labels_cheng);
    case {'wilcoxon', 'wilkcox'} %wilcoxan rank sum
        [score,index] = Wilkcoxnew(data, labels_cheng);
    case 'kstest'
        [score,index] = kstestnew(data, labels_cheng);
    case 'ftest'
        [score,index] = ftestnew(data, labels_cheng);
    case 'wilksnew'
        [score,index] = wilksnew(data, labels_cheng);
%     case 'mrmr'
%         dataw_discrete=makeDataDiscrete_mrmr(data);             
%         index= mrmr_mid_d(dataw_discrete, labels_cheng, size(data,2));
%         score=size(data,2):-1:1;
end
