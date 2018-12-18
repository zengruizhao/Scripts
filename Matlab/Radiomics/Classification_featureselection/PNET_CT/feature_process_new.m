function [ data, delete] = feature_process_new(Feature, phase)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    phase_feature = Feature(:, phase);
    numoffeature = 708;%765;%830;%240;
    delete = [];
    for i=1:size(phase_feature, 1)
       all_slice = phase_feature{i};
       temp = 0;
       feature = [];
       for slice=1:size(all_slice, 2)-1
           if ~isempty(all_slice{slice})
                temp = temp + 1;
                feature(temp, 1:numoffeature) = all_slice{slice};%(:, [1:690 756:830]);
           end
       end
       if isempty(feature)
          feature = zeros(1, numoffeature); 
          delete(end+1) = i;
       end
%        data(i, 1:numoffeature) = feature;
       data(i, 1:numoffeature) = mean(feature);%%
%        data(i, 1:numoffeature) = median(feature);
%        data(i, 1:numoffeature) = max(feature);%V_all
%        data(i, 1:numoffeature) = quantile(feature, 0.4);%%%
%        data(i, 1:numoffeature) = quantile(feature, 0.7);%% try
%%
%        data(i, 1:numoffeature) = mode(feature);
%        data(i, 1:numoffeature) = var(feature);
%        data(i, 1:numoffeature) = std(feature);
%        data(i, 1:numoffeature) = skewness(feature);
%        data(i, 1:numoffeature) = kurtosis(feature);
%        data(i, 1:numoffeature) = prctile(feature);
        data(i, numoffeature+1:numoffeature+130) = all_slice{slice + 1};
    end
end



