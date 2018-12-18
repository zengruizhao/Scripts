function feature = feature_process(Feature, phase)
%   Detailed explanation goes here
    phase_feature = Feature(:, phase);
    numoffeature = 1345;%1475;%260 125 960 130
    temp = 0;
    feature = zeros(size(phase_feature, 1), numoffeature);
    for i=1:size(phase_feature, 1)
       all_slice = phase_feature{i};
       temp = temp + 1;
       feature(temp, 1:numoffeature) = all_slice;
    end
end



