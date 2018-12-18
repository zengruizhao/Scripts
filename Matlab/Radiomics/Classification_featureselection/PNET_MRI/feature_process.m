function feature = feature_process(Feature)
%   Detailed explanation goes here
    numoffeature = 1475;%1475;%260 125 960 130
    temp = 0;
    feature = zeros(size(Feature, 1), numoffeature);
    for i=1:size(Feature, 1)
       all_slice = Feature{i};
       temp = temp + 1;
       feature(temp, 1:numoffeature) = all_slice;%(:, 1346:1475);% (1:260)(261:385)(386:1345)(1346:1475)
    end
end



