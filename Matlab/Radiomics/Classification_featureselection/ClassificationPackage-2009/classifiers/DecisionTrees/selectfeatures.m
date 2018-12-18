function datasubset = selectfeatures(data, dataset)

%% Reduce your data to a given subset of features

load tfeat
f = length(find(features(dataset,:)>0)); %fix length of unique genes
%%% finduniquegenes
datasubset(:,1:f) = data(:,features(dataset,1:f)); %%Read the unique genes to obtain subset