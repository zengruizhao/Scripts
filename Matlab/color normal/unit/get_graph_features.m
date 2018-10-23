function [vfeature,GraphFeatureDescription] = get_graph_features(x,y)

% graphfeats    Calculates graph-based features for nuclear centroids
% located at (x,y) in the image. 
% 
% Necessary input:
% x,y: x and y coordinates of points that will be used for graph construction
% (i.e. nuclear centroids). 

% Output Description: vfeature contains the following:

% Voronoi Features
% 1: Area Standard Deviation
% 2: Area Average
% 3: Area Minimum / Maximum
% 4: Area Disorder
% 5: Perimeter Standard Deviation
% 6: Perimeter Average
% 7: Perimeter Minimum / Maximum
% 8: Perimeter Disorder
% 9: Chord Standard Deviation
% 10: Chord Average
% 11: Chord Minimum / Maximum
% 12: Chord Disorder

% Delaunay Triangulation
% 13: Side Length Minimum / Maximum
% 14: Side Length Standard Deviation
% 15: Side Length Average
% 16: Side Length Disorder
% 17: Triangle Area Minimum / Maximum
% 18: Triangle Area Standard Deviation
% 19: Triangle Area Average
% 20: Triangle Area Disorder

% Minimum Spanning Tree
% 21: MST Edge Length Average
% 22: MST Edge Length Standard Deviation
% 23: MST Edge Length Minimum / Maximum
% 24: MST Edge Length Disorder

% Nuclear Features
% 25: Area of polygons
% 26: Number of nuclei
% 27: Density of Nuclei
% 28: Average distance to 3 Nearest Neighbors
% 29: Average distance to 5 Nearest Neighbors
% 30: Average distance to 7 Nearest Neighbors
% 31: Standard Deviation distance to 3 Nearest Neighbors
% 32: Standard Deviation distance to 5 Nearest Neighbors
% 33: Standard Deviation distance to 7 Nearest Neighbors
% 34: Disorder of distance to 3 Nearest Neighbors
% 35: Disorder of distance to 5 Nearest Neighbors
% 36: Disorder of distance to 7 Nearest Neighbors
% 37: Avg. Nearest Neighbors in a 10 Pixel Radius
% 38: Avg. Nearest Neighbors in a 20 Pixel Radius
% 39: Avg. Nearest Neighbors in a 30 Pixel Radius
% 40: Avg. Nearest Neighbors in a 40 Pixel Radius
% 41: Avg. Nearest Neighbors in a 50 Pixel Radius
% 42: Standard Deviation Nearest Neighbors in a 10 Pixel Radius
% 43: Standard Deviation Nearest Neighbors in a 20 Pixel Radius
% 44: Standard Deviation Nearest Neighbors in a 30 Pixel Radius
% 45: Standard Deviation Nearest Neighbors in a 40 Pixel Radius
% 46: Standard Deviation Nearest Neighbors in a 50 Pixel Radius
% 47: Disorder of Nearest Neighbors in a 10 Pixel Radius
% 48: Disorder of Nearest Neighbors in a 20 Pixel Radius
% 49: Disorder of Nearest Neighbors in a 30 Pixel Radius
% 50: Disorder of Nearest Neighbors in a 40 Pixel Radius
% 51: Disorder of Nearest Neighbors in a 50 Pixel Radius

load('GraphFeatureDescription.mat')

% Calculate the Voronoi diagram.
[VX,VY] = voronoi(x,y);
[V, C] = voronoin([x(:),y(:)]);

% Okay, so:
% VX, VY - These guys contain the vertices in a way such that
% plot(VX,VY,'-',x,y,'.') creates the voronoi diagram. I don't actually
% think these are used later on, but I'm keeping them here just in case.

% C - This variable is an m by 1 cell array, where m is the number of cell
% centroids in your image. Each element in C is a vector
% with the coordinates of the vertices of that row's voronoi polygon. 
% V - This is a q by 2 matrix, where q is the number of vertices and 2 is
% the number of dimensions of your image. Each element in V contains the
% location of the vertex in 2D space.
% The idea here is that if you want to see the coordinates for the vertices
% of polygon 5, for example, you would go:
%     X = V(C{5},:)
% which would display five rows, each with the 2D coordinates of the vertex
% of polygon 5.

% Get the delaunay triangulation...
del = delaunay(x,y);    

% Returns a set of triangles such that no data points are contained in any 
% triangle's circumcircle. Each row of the numt-by-3 matrix TRI defines 
% one such triangle and contains indices into the vectors X and Y. When 
% the triangles cannot be computed (such as when the original data is 
% collinear, or X is empty), an empty matrix is returned.
    
% Get the Minimum Spanning Tree (MST) (optional: plot)
[mst.edges mst.edgelen mst.totlen] = mstree([x,y],[],0);

% Record indices of inf and extreme values to skip these cells later
Vnew        = V;
Vnew(1,:)   = [];

% Find the data points that lie far outside the range of the data
[Vsorted,I]     = sort([Vnew(:,1);Vnew(:,2)]);
N               = length(Vsorted);
Q1              = round(0.25*(N+1));
Q3              = round(0.75*(N+1));
IQR             = Q3 - Q1;
highrange       = Q3 + 1.5*IQR;
lowrange        = Q1 - 1.5*IQR;
Vextreme        = [];
Vextreme        = [Vextreme; V(find(V > highrange))];
Vextreme        = [Vextreme; V(find(V < lowrange))];

banned = [];
for i = 1:length(C)
    if(~isempty(C{i}))
        
    if(max(any(isinf(V(C{i},:)))) == 1 || max(max(ismember(V(C{i},:),Vextreme))) == 1)
        banned = [banned, i];
    end
    end
end
% If you've eliminated the whole thing (or most of it), then only ban 
% indices that are infinity (leave the outliers)
if(length(banned) > length(C)-2)
    banned = [];
    for i = 1:length(C)
        if(max(any(isinf(V(C{i},:)))) == 1)
            banned = [banned, i];
        end
    end
end

% Voronoi Diagram Features
% Area
c = 1;
d = 1;
e = d;
for i = 1:length(C)

    if(~ismember(i,banned) && ~isempty(C{i}))
        X = V(C{i},:);
        chord(1,:) = X(:,1);
        chord(2,:) = X(:,2);
        % Calculate the chord lengths (each point to each other point)
        for ii = 1:size(chord,2)
            for jj = ii+1:size(chord,2)
                chorddist(d) = sqrt((chord(1,ii) - chord(1,jj))^2 + (chord(2,ii) - chord(2,jj))^2);
                d = d + 1;
            end
        end

        % Calculate perimeter distance (each point to each nearby point)
        for ii = 1:size(X,1)-1
            perimdist(e) = sqrt((X(ii,1) - X(ii+1,1))^2 + (X(ii,2) - X(ii+1,2))^2);
            e = e + 1;
        end
        perimdist(size(X,1)) = sqrt((X(size(X,1),1) - X(1,1))^2 + (X(size(X,1),2) - X(1,2))^2);
        
        % Calculate the area of the polygon
        area(c) = polyarea(X(:,1),X(:,2));
        c = c + 1;
        clear chord X
    end
end
if(~exist('area','var'))
    vfeature = zeros(1,51);
    return; 
end
vfeature(1) = std(area); 
vfeature(2) = mean(area);
vfeature(3) = min(area) / max(area);
vfeature(4) = 1 - ( 1 / (1 + (vfeature(1) / vfeature(2))) );

vfeature(5) = std(perimdist);
vfeature(6) = mean(perimdist);
vfeature(7) = min(perimdist) / max(perimdist);
vfeature(8) = 1 - ( 1 / (1 + (vfeature(5) / vfeature(6))) );

vfeature(9) = std(chorddist);
vfeature(10) = mean(chorddist);
vfeature(11) = min(chorddist) / max(chorddist);
vfeature(12) = 1 - ( 1 / (1 + (vfeature(9) / vfeature(10))) );

% Delaunay 
% Edge length and area
c = 1;
d = 1;
for i = 1:size(del,1)
    t = [x(del(i,:)),y(del(i,:))];
    
    sidelen(c:c+2) = [sqrt( ( t(1,1) - t(2,1) )^2 + (t(1,2) - t(2,2))^2 ), ...
        sqrt( ( t(1,1) - t(3,1) )^2 + (t(1,2) - t(3,2))^2 ), ...
        sqrt( ( t(2,1) - t(3,1) )^2 + (t(2,2) - t(3,2))^2 )];
    dis(i,1:3) = sum( sidelen(c:c+2) );
    c = c + 3;
    triarea(d) = polyarea(t(:,1),t(:,2));
    d = d + 1;
end

vfeature(13) = min(sidelen) / max(sidelen);
vfeature(14) = std(sidelen); 
vfeature(15) = mean(sidelen);
vfeature(16) = 1 - (1 / (1 + (vfeature(14) / vfeature(15)) ) );

vfeature(17) = min(triarea) / max(triarea);
vfeature(18) = std(triarea);
vfeature(19) = mean(triarea);
vfeature(20) = 1 - (1 / (1 + (vfeature(18) / vfeature(19))) );


% MST: Average MST Edge Length
% The MST is a tree that spans the entire population in such a way that the
% sum of the Euclidian edge length is minimal.

vfeature(21) = mean(mst.edgelen);
vfeature(22) = std(mst.edgelen);
vfeature(23) = min(mst.edgelen) / max(mst.edgelen);
vfeature(24) = 1 - ( 1 / ( 1 + (vfeature(22)/vfeature(21)) ) ); 

% Nuclear Features
% Density
vfeature(25) = sum(area); 
vfeature(26) = size(C,1);
vfeature(27) = vfeature(26) / vfeature(25);

% Average Distance to K-NN
% Construct N x N distance matrix:
for i = 1:size(x,1)
    for j = 1:size(x,1)
        distmat(i,j) = sqrt( (x(i) - x(j))^2 + (y(i) - y(j))^2 );
    end
end
DKNN = zeros(3,size(distmat,1));
kcount = 1;
for K = [3,5,7]
    % Calculate the summed distance of each point to it's K nearest neighbours

    for i = 1:size(distmat,1)
        tmp = sort(distmat(i,:),'ascend');

        % NOTE: when finding the summed distance, throw out the first result,
        % since it's the zero value at distmat(x,x). Add 1 to K to compensate.
        DKNN(kcount,i) = sum(tmp(2:K+1));
    end
    kcount = kcount + 1;
end
vfeature(28) = mean(DKNN(1,:));
vfeature(29) = mean(DKNN(2,:));
vfeature(30) = mean(DKNN(3,:));

vfeature(31) = std(DKNN(1,:));
vfeature(32) = std(DKNN(2,:));
vfeature(33) = std(DKNN(3,:));

vfeature(34) = 1 - (1 / ( 1 + (vfeature(31) / (vfeature(28)+eps)) ));
vfeature(35) = 1 - (1 / ( 1 + (vfeature(32) / (vfeature(29)+eps)) ));
vfeature(36) = 1 - (1 / ( 1 + (vfeature(33) / (vfeature(30)+eps)) ));

% NNRR_av: Average Number of Neighbors in a Restricted Radius
% Set the number of pixels within which to search
rcount = 1;
for R = [10:10:50]

    % For each point, find the number of neighbors within R pixels
    for i = 1:size(distmat,1)

        % NOTE: as above with the K-NN calculation, we subtract 1 from the
        % number of pixels found, because this corresponds to the diagonal of
        % the distance matrix, which is always 0 (i.e. distmat(x,x) = 0 for all
        % x).
        NNRR(rcount,i) = length( find( distmat(i,:) <= R ) ) - 1;
    end
    if(sum(NNRR(rcount,:)) == 0)
        eval(['NNRR_av_' num2str(R) ' = 0;']);
        eval(['NNRR_sd_' num2str(R) ' = 0;']);
        eval(['NNRR_dis_' num2str(R) ' = 0;']);
    else
        eval(['NNRR_av_' num2str(R) ' = mean(NNRR(rcount,:));']);
        eval(['NNRR_sd_' num2str(R) ' = std(NNRR(rcount,:));']);
        eval(['NNRR_dis_' num2str(R) ' = 1 - (1 / (1 + (NNRR_sd_' num2str(R) '/NNRR_av_' num2str(R) ')));']);
    end
    
    rcount = rcount + 1;
end

vfeature(37) = NNRR_av_10;
vfeature(38) = NNRR_av_20;
vfeature(39) = NNRR_av_30;
vfeature(40) = NNRR_av_40;
vfeature(41) = NNRR_av_50;
vfeature(42) = NNRR_sd_10;
vfeature(43) = NNRR_sd_20;
vfeature(44) = NNRR_sd_30;
vfeature(45) = NNRR_sd_40;
vfeature(46) = NNRR_sd_50;
vfeature(47) = NNRR_dis_10;
vfeature(48) = NNRR_dis_20;
vfeature(49) = NNRR_dis_30;
vfeature(50) = NNRR_dis_40;
vfeature(51) = NNRR_dis_50;






  


  
  
 

