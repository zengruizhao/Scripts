function [meanAcc stdAcc] = parseClassificationStats(STATS)

acc = [];
for i=1:length(STATS)
    acc = [acc STATS{i}.acc];
end

meanAcc = mean(acc);
stdAcc = std(acc);