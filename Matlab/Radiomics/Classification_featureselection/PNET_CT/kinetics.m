function [ kinetics_feature ] = kinetics( Feature )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    numoffeature = 272;Delete = [];
    for phase=1:4
        [kinetics{phase}, delete]= feature_process(Feature, phase);
        if ~isempty(delete)
            Delete(end+1) = delete;
        end
    end
    for phase=1:4
        kinetics{phase}(Delete, :) = [];
    end
    x=[0, 1, 2, 3];
    for feature=1:numoffeature
        x1 = kinetics{1}(:, feature);
        x2 = kinetics{2}(:, feature);
        x3 = kinetics{3}(:, feature);
        x4 = kinetics{4}(:, feature);
        for case_file =1:size(x1, 1)
          y=[x1(case_file), x2(case_file),x3(case_file),x4(case_file)];
          plot(x, y);
          hold on;
        end
    end
end

