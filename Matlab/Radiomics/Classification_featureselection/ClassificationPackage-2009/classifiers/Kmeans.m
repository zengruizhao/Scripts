function [methodstring,tp,tn,fp,fn,prediction] = Kmeans(data,labels)
% Classification via K-meansClustering
% [tp,tn,fp,fn]=Kmeans(data,labels,kcluster)
% 
% (c) George Lee (2008)

methodstring = 'k-means classification ';

%%% We are hard coding these parameters %%%
isRand = 1;
kcluster = length(unique(labels));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if nargin < 4
%     isRand = 0; % Default - Not Random, 1 for random.
%     if nargin < 3
%         kcluster = 2;
%     end
% end

g = kmeanscluster(data,kcluster,isRand); %run k-means algorithm->output new labels

%Calculate Accuracy
g = ((g - 1)*2)-1; %format decision labels
labels = ((labels - 1)*2)-1; %format ground truth labels

prediction=g;
[tp,tn,fp,fn] = count_values(g,labels); %count values
