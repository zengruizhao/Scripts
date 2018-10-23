clear;clc;
load datafile_new;
load optimal_num_of_feature;
data=data(:,1:60);
algorithms = {'mim'
              'mrmr'
              'cmim'
              'jmi'
              'disr'
              'cife'
              'icap'
              'condred'};
for num=1:(length(algorithms))
    selectedFeature(num,:)= (feast(algorithms(num),20,data,label))';
    disp([num algorithms(num)]);
end 

% [selectedFeatures, featureScores] = feast('relief',10,data,label);
[m, n] = size(data);
for j=1:m %%select optimal features
    for i=1:length(a)
        new_data(j,i)= data(j, a(i));
    end
end