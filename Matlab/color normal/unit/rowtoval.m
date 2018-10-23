% Rowtoval: Converts rows of a matrix (numeric or character) to a vector of 
%           scalars.  Useful for checking for identical rows or for sorting 
%           rows into lexicological sequence [but see sortrows() for the latter].
%
%     Usage: [vals,base] = rowtoval(x,{base})
%
%           x =     numeric or character matrix.
%           base =  optional base for transformation; useful for finding values
%                     for rows not included in a previous transformation.
%           -------------------------------------------------------------------
%           vals =  column vector of representative scalar values.
%

% RE Strauss, 8/31/99
%   3/22/02 - added option of optional base.
%   5/5/06 -  convert input matrix to numeric before processing.

function [vals,base] = rowtoval(x,base)
  if (nargin < 2) base = []; end;

  x = double(x);
  [r,c] = size(x);
  xmin = min(min(x));
  x = x - xmin;
  if (isempty(base))
    base = max(max(x))+1;
  end;

  vals = zeros(r,1);
  for i = 1:c
    vals = vals + x(:,i)*base^(c-i);
  end;

  return;