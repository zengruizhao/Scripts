%   AUTHOR:
%       Boguslaw Obara, http://boguslawobara.net/
%% Create a polygon
clear;clc;
img = imread('./2.png');
img = im2bw(img);
img = 1-img;
P = bwboundaries(img);

PP = P{1};
%% Plot
plot(PP(:,1),PP(:,2),'-bs'); hold on
%% Query
q = PP(10,:);
plot(q(:,1),q(:,2),'r*');hold on
stage = BOPointInPolygon(PP,q);
disp(['Stage: ' stage]);
for i=1:5
    [pointx, pointy] = ginput(1);    
    q = [pointx pointy];
    plot(q(:,1),q(:,2),'r*');hold on
    %q = P(13,:)
    stage = BOPointInPolygon(PP,q);
    disp(['Stage: ' stage]);
end
%%