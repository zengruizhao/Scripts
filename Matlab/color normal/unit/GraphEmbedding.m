function EigV = GraphEmbedding(X,newdim)

W = L2_distance(X', X', 1);
% W - represents  your distance map.
%   - a 21 X 21 matrix representing the distance
%   - of each patient from every other patient
%   - in the high dimensional space.
% data.one = 6; 
% data.two = 9+data.one; 
% data.three= 22+data.two; 
% data.four= 6+data.three; 
% data.five= 15+data.four ; 
% data.six= 27+data.five; 
% data.seven= 27+data.six;

%V = exp(-W/max(max(W)));

%figure; imagesc(V);
%title('Confusion matrix for tumor/non-tumor regions');

% Doing the graph-cut partitioning on the confusion matrix.
% EigV = ncut_b(V,4); % with transformation
 EigV  = ncut(W,newdim+1); % without transformation
%EigV = exp(-EigV/max(max(EigV)));
clear('W');
%EigV = EigV(:,2:5);


function [v,d] = ncut(A,nv)
%
%  [v,d] = ncut(A,nv)
%
%
%    computes 'nv' of the normalized cut vectors 'v' from
%    matrix 'A'
%
%    it computes the largest eigenvectors of
%      A*v = \lambda D * v;   D = diag(sum(A));
%
%    this is same as solving the smallest eigenvectors of
%     (D-A)*v = \lambda D *v;
%   

%
%  Jianbo Shi   
%

ds = sum(A);
ds = ones(size(ds))./sqrt(ds);

for j=1:size(A,1),
  A(j,:) = A(j,:).*ds;
end

for j=1:size(A,2);
  A(:,j) = A(:,j).*ds';
end


%disp(sprintf('computing eig values'));
OPTIONS.tol=1e-10;
OPTIONS.maxit=15;
OPTIONS.disp=0;
%tic;toc;

[v,d] = eigs(A,nv,'LM',OPTIONS);

d = abs(diag(d));

for j=1:nv,
  v(:,j) = v(:,j).*ds';
end

%% George Lee
v=v(:,2:nv);%adjust for dummy first eigenvector

function d = L2_distance(a,b,df)
% L2_DISTANCE - computes Euclidean distance matrix
%
% E = L2_distance(A,B)
%
%    A - (DxM) matrix 
%    B - (DxN) matrix
%    df = 1, force diagonals to be zero; 0 (default), do not force
% 
% Returns:
%    E - (MxN) Euclidean distances between vectors in A and B
%
%
% Description : 
%    This fully vectorized (VERY FAST!) m-file computes the 
%    Euclidean distance between two vectors by:
%
%                 ||A-B|| = sqrt ( ||A||^2 + ||B||^2 - 2*A.B )
%
% Example : 
%    A = rand(400,100); B = rand(400,200);
%    d = distance(A,B);

% Author   : Roland Bunschoten
%            University of Amsterdam
%            Intelligent Autonomous Systems (IAS) group
%            Kruislaan 403  1098 SJ Amsterdam
%            tel.(+31)20-5257524
%            bunschot@wins.uva.nl
% Last Rev : Wed Oct 20 08:58:08 MET DST 1999
% Tested   : PC Matlab v5.2 and Solaris Matlab v5.3

% Copyright notice: You are free to modify, extend and distribute 
%    this code granted that the author of the original code is 
%    mentioned as the original author of the code.

% Fixed by JBT (3/18/00) to work for 1-dimensional vectors
% and to warn for imaginary numbers.  Also ensures that 
% output is all real, and allows the option of forcing diagonals to
% be zero.  

if (nargin < 2)
   error('Not enough input arguments');
end

if (nargin < 3)
   df = 0;    % by default, do not force 0 on the diagonal
end

if (size(a,1) ~= size(b,1))
   error('A and B should be of same dimensionality');
end

if ~(isreal(a)*isreal(b))
   disp('Warning: running distance.m with imaginary numbers.  Results may be off.'); 
end

if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end

aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
d = sqrt(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

% make sure result is all real
d = real(d); 

% force 0 on the diagonal? 
if (df==1)
  d = d.*(1-eye(size(d)));
end


