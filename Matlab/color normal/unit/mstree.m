%--------------------------------------------------------------------------
% MSTREE: Minimum spanning tree
%
%     Syntax: [edges,edgelen,totlen] = mstree(crds,{labels},{doplot})
%
%         crds =    [N x p] matrix of point coordinates.
%         labels =  optional [N x 1] vector of numeric or character labels.
%         doplot =  optional boolean flag indicating whether (=1) or not (=0) 
%                     to plot the points and minimum spanning tree in the space 
%                     of the first two axes [default = 0].
%         ---------------------------------------------------------------------
%         edges =   [(N-1) x 2] list of points defining edges.
%         edgelen = [(N-1) x 1] vector of edge lengths corresponding to
%                     'edges'.
%         totlen =  total length of tree (sum of edge lengths).
%

% RE Strauss, 7/9/96
%   2/9/99 -   produces plot only for 2D coordinates, but will find the tree
%                for any number of dimensions.
%   9/7/99 -   changed plot colors for Matlab v5.
%   10/8/03 -  added point numbers to plot.
%   10/14/03 - sort edges by increasing point id.
%   12/5/03 -  if p>2, produce plot for first two axes; suppress plot for p=1;
%                use rowtoval() to sort edges.
%   12/9/03 -  added optional plot labels.

function [edges,edgelen,totlen] = mstree(crds,labels,doplot)
  if (nargin < 2) labels = []; end;
  if (nargin < 3) doplot = []; end;

  [n,p] = size(crds);

  if (isempty(doplot) | p<2)
    doplot = 0;
  end;
  
  if (~isempty(labels))
    if (~ischar(labels))
      labels = tostr(labels(:));
    end;
  end;

  totlen = 0;
  edges = zeros(n-1,2);
  edgelen = zeros(n-1,1);

  dist = eucl(crds);            % Pairwise euclidean distances
  highval = max(max(dist))+1;


  e1 = 1*ones(n-1,1);           % Initialize potential-edge list
  e2 = [2:n]';
  ed =  dist(2:n,1);

  for edge = 1:(n-1)            % Find shortest edges
    [mindist,i] = min(ed);        % New edge
    t = e1(i);
    u = e2(i);
    totlen = totlen + mindist;
    if (t<u)                      % Add to actual-edge list
      edges(edge,:) = [t u];
    else
      edges(edge,:) = [u t];
    end;
    edgelen(edge) = mindist;

    if (edge < n-1)
      i = find(e2==u);              % Remove new vertex from
      e1(i) = 0;                    %   potential-edge list
      e2(i) = 0;
      ed(i) = highval;

      indx = find(e1>0);
      for i = 1:length(indx)        % Update potential-edge list
        j = indx(i);
        t = e1(j);
        v = e2(j);
        if (dist(u,v) < dist(t,v))
          e1(j) = u;
          ed(j) = dist(u,v);
        end;
      end;
    end;
  end;
  
  v = rowtoval(edges);
  [v,edges,edgelen] = sortmat(v,edges,edgelen);  % Sort by increasing pt identifier

  if (doplot)

%     plot(crds(:,1),crds(:,2),'ok');
    putbnd(crds(:,1),crds(:,2));
    
    deltax = 0.018*range(crds(:,1));
    deltay = 0.02*range(crds(:,2));
    for i = 1:n
      if (isempty(labels))
        lab = int2str(i);
      else
        lab = labels(i,:);
      end;
%       text(crds(i,1)+deltax,crds(i,2)+deltay,lab);
    end;
    
    hold on;
    for i = 1:(n-1)
      t = edges(i,1);
      u = edges(i,2);
      x = [crds(t,1); crds(u,1)];
      y = [crds(t,2); crds(u,2)];
      plot(x,y,'-b');
    end;
    hold off;
  end;

  return;