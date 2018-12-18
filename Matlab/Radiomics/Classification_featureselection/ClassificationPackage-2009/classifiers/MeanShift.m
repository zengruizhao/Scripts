function [methodstring,tp,tn,fp,fn,point2cluster] = MeanShift(x,bandwidth)

% Runs Meanshift
% run_meanshift(x,bandwidth)
% x - data
% bandwitch - window size

%% Generate Data

% nPtsPerClust = 250;
% nClust  = 3;
% totalNumPts = nPtsPerClust*nClust;
% m(:,1) = [1 1]';
% m(:,2) = [-1 -1]';
% m(:,3) = [1 -1]';
% var = .6;
% bandwidth = .75;
% clustMed = [];
%clustCent;
% x = var*randn(2,nPtsPerClust*nClust);
% %*** build the point set
% for i = 1:nClust
%     x(:,1+(i-1)*nPtsPerClust:(i)*nPtsPerClust)       = x(:,1+(i-1)*nPtsPerClust:(i)*nPtsPerClust) + repmat(m(:,i),1,nPtsPerClust);   
% end

%%% added by ANB %%%
x = x';
methodstring = 'Mean Shift classifier ';
tp=[]; tn=[]; fp=[]; fn=[]; % not really sure where this info is?
%%%%%%%%%%%%%%%%%%%%

[point2cluster,clustMembsCell,clustCent] = MeanShiftCluster(x,bandwidth);

numClust = length(clustMembsCell);

figure(10),clf,hold on
cVec = 'bgrcmykbgrcmykbgrcmykbgrcmyk';%, cVec = [cVec cVec];
for k = 1:min(numClust,length(cVec))
    myMembers = clustMembsCell{k}; % points
    plot(x(1,myMembers),x(2,myMembers),[cVec(k) '.'])

%     myClustCen = clustCent(:,k); % cluster centroids
%     plot(myClustCen(1),myClustCen(2),'o','MarkerEdgeColor','k','MarkerFaceColor',cVec(k), 'MarkerSize',10); 
end
title(['no shifting, numClust:' int2str(numClust) 'Bandwidth: ' num2str(bandwidth)])