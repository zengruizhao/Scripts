function feature = feature_process_new_1(Feature, phase)
%   Detailed explanation goes here
    phase_feature = Feature(:, phase);
    numoffeature = 385;%520;%385;%765;%830;%240;
    temp = 0;
    for i=1:size(phase_feature, 1)
       all_slice = phase_feature{i};
       temp = temp + 1;
       feature(temp, 1:numoffeature) = all_slice;%(:, [1:690 756:830]);  
    end
end



