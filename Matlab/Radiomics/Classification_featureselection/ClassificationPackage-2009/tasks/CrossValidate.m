function [stats] = CrossValidate(type,varargin)

stats = feval(type,varargin{:});