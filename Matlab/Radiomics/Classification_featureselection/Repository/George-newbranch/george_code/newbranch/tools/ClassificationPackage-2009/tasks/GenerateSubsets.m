function [training testing] = GenerateSubsets(type, varargin)

try
    [training testing] = feval(type,varargin{:});
catch e
    fprintf('May include invalid method.\n');
    rethrow(e);
end