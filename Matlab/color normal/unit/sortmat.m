 % SORTMAT:  Sorts a primary "key" column vector, and re-sequences the rows of 
%           one or more secondary matrices into the corresponding sequence.
%
%     Usage: [outvect,outmat1,...,outmat9] = sortmat(invect,inmat1,{inmat2},...,{inmat9})
%
%           invect = primary (column) vector to be sorted.
%           inmat1 = first secondary matrix to be resequenced (can be null).
%           inmat2 = optional second matrix to be resequenced (can be null).
%             .
%             .
%             .
%           inmat9 = optional ninth matrix to be resequenced (can be null).
%           --------------------------------------------------------
%           outvect = sorted primary vector.
%           outmat1 = resequenced first matrix.
%             .
%             .
%             .
%           outmat9 = resequenced ninth matrix.
%

function [outvect,outmat1,outmat2,outmat3,outmat4,outmat5,outmat6,outmat7,outmat8,outmat9] ...
            = sortmat(invect,inmat1,inmat2,inmat3,inmat4,inmat5,inmat6,inmat7,inmat8,inmat9)

  if (nargin ~= nargout)
    error('SORTMAT: number of output arguments must match number of input arguments');
  end;

  [r,c] = size(invect);
  if (c>1)
    if (r>1)
      error('SORTMAT: first input argument must be a column vector');
    else
      invect = invect';
      r = c;
    end;
  end;

  outmat1 = [];
  outmat2 = [];
  outmat3 = [];
  outmat4 = [];
  outmat5 = [];
  outmat6 = [];
  outmat7 = [];
  outmat8 = [];
  outmat9 = [];

  [outvect,seq] = sort(invect);

  if (nargin > 1)
    if (~isempty(inmat1))
      if (size(inmat1,1) ~= r)
        error('SORTMAT: <inmat1> must have same number of rows as <invect>');
      end;
      outmat1 = inmat1(seq,:);
    end;
  end;

  if (nargin > 2)
    if (~isempty(inmat2))
      if (size(inmat2,1) ~= r)
        error('SORTMAT: <inmat2> must have same number of rows as <invect>');
      end;
      outmat2 = inmat2(seq,:);
    end;
  end;

  if (nargin > 3)
    if (~isempty(inmat3))
      if (size(inmat3,1) ~= r)
        error('SORTMAT: <inmat3> must have same number of rows as <invect>');
      end;
      outmat3 = inmat3(seq,:);
    end;
  end;

  if (nargin > 4)
    if (~isempty(inmat4))
      if (size(inmat4,1) ~= r)
        error('SORTMAT: <inmat4> must have same number of rows as <invect>');
      end;
      outmat4 = inmat4(seq,:);
    end;
  end;

  if (nargin > 5)
    if (~isempty(inmat5))
      if (size(inmat5,1) ~= r)
        error('SORTMAT: <inmat5> must have same number of rows as <invect>');
      end;
      outmat5 = inmat5(seq,:);
    end;
  end;

  if (nargin > 6)
    if (~isempty(inmat6))
      if (size(inmat6,1) ~= r)
        error('SORTMAT: <inmat6> must have same number of rows as <invect>');
      end;
      outmat6 = inmat6(seq,:);
    end;
  end;

  if (nargin > 7)
    if (~isempty(inmat7))
      if (size(inmat7,1) ~= r)
        error('SORTMAT: <inmat7> must have same number of rows as <invect>');
      end;
      outmat7 = inmat7(seq,:);
    end;
  end;

  if (nargin > 8)
    if (~isempty(inmat8))
      if (size(inmat8,1) ~= r)
        error('SORTMAT: <inmat8> must have same number of rows as <invect>');
      end;
      outmat8 = inmat8(seq,:);
    end;
  end;

  if (nargin > 9)
    if (~isempty(inmat9))
      if (size(inmat9,1) ~= r)
        error('SORTMAT: <inmat9> must have same number of rows as <invect>');
      end;
      outmat9 = inmat9(seq,:);
    end;
  end;

  return;
