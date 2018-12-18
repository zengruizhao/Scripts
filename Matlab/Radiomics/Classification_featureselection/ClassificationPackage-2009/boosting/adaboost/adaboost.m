function [alpha, bestclass, error, h] = adaboost(training, labels, varargin)
% AdaBoost Algorithm. Input "training" consists of probabilities ranging 
% from [0,1]. Both T and "debug" are optional. Default is T=10, debug=0.
% 
% Scott Doyle, 02/24/2009 
% Ver. 1.0 - Rewritten the AdaBoost code, mostly from the ground up.
% Scott Doyle, 09/16/2010
% Ver. 2.0 - Cleaned up the code, trying to get all versions consolidated.

% First check the input arguments and set defaults appropriately.
if(nargin == 2)
    T       = 10;
    debug   = 0;
elseif(nargin == 3)
    T       = varargin{1};
    debug   = 0;
elseif(nargin == 4)
    T       = varargin{1};
    debug   = varargin{2};
end

% Assume the data is arranged as rows = examples, cols = features. Check to
% make sure nothing funny is going on with the label / data sizes.
[n m]   = size(training);
l       = length(labels);

if(l ~= n)
    fprintf('\nError: number of labels does not equal number of objects. Aborting...\n');
    return;
end

if(numel(unique(labels)) ~= 2)
    fprintf('\nError: number of labels is greater than 2. Aborting...\n');
    return;
end

% Force labels to be 0/1
labels(labels == min(labels)) = 0;
labels(labels == max(labels)) = 1;

l_pos   = length(find(labels == max(labels)));
l_neg   = length(find(labels == min(labels)));

% Print some stuff out if we're debugging
if(debug == 1)
    fprintf('\n');
    fprintf('=============================\n');
    fprintf('Beginning AdaBoost Algorithm \n');
    fprintf('=============================\n');
    fprintf('Parameters:\n');
    fprintf('\tT:\t\t%d\n',T);
    fprintf('\tObjects:\t%d\n',n);
    fprintf('\tFeatures:\t%d\n',m);
    fprintf('\n');
end

% Set up the weight variable (row = iteration, col = sample). 
w = zeros(T,n);

% Initialize weights to 1/n.
w(1,:) = 1/(n);

% Set up some output variables.
error   = zeros(T,1);
h       = zeros(T,1);
beta    = zeros(T,1);
alpha   = zeros(T,1);
bestclass = zeros(n,T);

% Begin the Adaboost iterations
for t=1:T
    if(debug == 1), fprintf('Beginning Iteration %d...\n',t); end;
    
    % Zero out the threshold and epsilon parameters
    thresh = zeros(m,1);
    epsilon = zeros(m,1);
    
    % Cycle through each feature.
    for j=1:m
        
        this_feature = training(:,j);
        
        % We are assuming that each column in the training represents the
        % PROBABILITIES of that feature. If this is not true, then this
        % whole section will be weird. 
        
%         % Using simple Otsu thresholding on the probabilies, in case 
%         % they're scaled weirdly. You can change this section by deleting
%         % the thresholding stuff and uncommenting the simple >< 0.5 if
%         % you're certain that the probabilies are centered around the
%         % middle.
%         thresh(j) = graythresh(this_feature);
%         this_result = im2bw(this_feature,thresh(j));
        
        k = isa(this_feature,'logical');
        if k == 0
            this_result = this_feature >= 0.5;
        else
            this_result = this_feature;
        end
        
        % Check to make sure there isn't something odd with the features or
        % the result. This might not need to be calculated anymore, since
        % it should be clear that this_feature should be between 0 and 1,
        % and this_result should be just a binary output.
        if(any(isnan(this_result)) || any(this_feature == -1))
            epsilon(j) = 9999999;
        else
            % Calculate the error (weighted by w)
%             epsilon(j) = sum(w(t,find(this_result~=labels)));
            epsilon(j) = sum(w(t,this_result~=labels));
        end
        
    end
    
    % Min epsilon = best feature, so select it and its error
    [error(t) h(t)] = min(epsilon);
    
    % Get the classification using best feature. If you uncommented the
    % alternative "this_result" calculation, also uncomment this one. 
%    bestclass(:,t) = im2bw(training(:,h(t)), thresh(h(t)));
     bestclass(:,t) = training(:,h(t)) >= 0.5;
    
    % Get alpha value
    beta(t) = error(t)/(1-error(t));
    alpha(t) = 0.5*log(1/(beta(t)+eps)); 
    
    % Calculate updated w's based on this feature
    if t+1 <= T
        for ii = 1:n
            % Each sample position maintains a specific weight. Thus, here
            % we cycle through each of the samples -- if it was correctly
            % classified, it gets one weight; if it's incorrectly
            % classified, it gets another. 
            if(labels(ii) == bestclass(ii,t))
                w(t+1,ii) = w(t,ii) * exp(-alpha(t) );
            else
                w(t+1,ii) = w(t,ii) * exp( alpha(t) );
            end
        end
        
        % Finally we add a scaling parameter (some AdaBoost papers have
        % this as Z_t) to make sure that the weights for the next feature
        % remain a distribution. 
        w(t+1,:) = w(t+1,:)./sum(w(t+1,:));
    end
    
end