function [stats, methodstring] = Classify(type, varargin)

% if ~isunix
%     error('Classification unavailable on Windows OS.  Please use Linux OS.');
% else

if ismember(lower(type),{'baggedc45','svm','onevsone_baggedc45','baggedc45_notraining','baggedc45_notraining_multiclass','baggedc45_multiclass','nbayes','nbayes_notraining','qda','lda'})
    [methodstring,stats]=feval(type, varargin{:});
else
    [methodstring,stats.tp,stats.tn,stats.fp,stats.fn,stats.prediction]=feval(type, varargin{:});
    stats.acc = (stats.tp+stats.tn)/(stats.tp+stats.tn+stats.fp+stats.fn);
    stats.ppv = stats.tp/(stats.tp+stats.fp);
    stats.sens = stats.tp/(stats.tp+stats.fn);
    stats.spec = stats.tn/(stats.fp+stats.tn);
end
% end
