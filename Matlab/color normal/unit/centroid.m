% 获取区域中心坐标点
% centroid.m  calculates the area and the coordinates of the centroid of
%             a polygonal region whose vertices are given
%             计算一个给出顶点多边形区域的面积和图心
%             by their rectangular coordinates.
% *****>      Input:
%               XY, an n-by-2 matrix whose rows are 行是逆时针方向列出的顶点坐标
%             counterclockwise listed coordinates of the
%             vertices.
% *****>      Output: CX, CY - coordinates of the centroid
%                          A - area of the region
%
% *****>      Call: [CX CY A]=centroid(XY)
%
%             Method: (position)=(static moment)/(area)
%                    transformed using Green's formula
%             (x-moment)=(area integral)(x dA)
%                       =(boundary integral)(-xy dx)
%             similarly (up to sign) for y-moment;
%             (area)=(1/2)*(boundary integral)(x dy - y dx)
%             evaluated along the boundary edges and
%             re-arranged for greater symmetry
%             By Z.V. Kovarik (kovarik@mcmaster.ca), May 1996

function [CX,CY,A]=centroid(XY);
n = size(XY);
n = n(1);
ST = XY([2:n,1],:);                             % cyclic shift of XY
UV = ST+XY;
ST = ST-XY;                            % sums and differences

CXY = ((3*prod(UV')+prod(ST'))*ST)/12;          % x- and y-moments
A = sum(UV(:,1).*ST(:,2)-UV(:,2).*ST(:,1))/4;   % area
CXY = CXY/A;
CX = -round(CXY(1));CY=round(CXY(2));               % centroid coordinates 

A = abs(A);