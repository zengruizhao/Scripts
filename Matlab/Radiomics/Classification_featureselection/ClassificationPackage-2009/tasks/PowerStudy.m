function PowerStudy(data, labels,T1,T2,n,tra_in,tes_in)
% Ajay Basavanhally
% Implemented as found in Mukherjee, et al., Estimating Dataset Size
% Requirements for Classifying DNA Microarray Data, 2003.

M = size(data,1);
labels = labels(:);
trainFlag = true;

%check inputs and use default values as needed
if nargin < 7 || isempty(tes_in) || nargin < 6 || isempty(tra_in)
    trainFlag = true;
else
    trainFlag = false;
end
if nargin < 5 || isempty(n)
    n = round(linspace(10,M-10,10)); % automatically select training set sizes
end
if nargin < 4 || isempty(T2)
    T2 = 50;
end
if nargin < 3 || isempty(T1)
    T1 = 50;
end

if size(labels,1) ~= M
    error('Features and labels are different sizes!');
end
if M < 10
    error('At least 10 samples are required');
end

%%% Part A: Subsampling Test %%%
%n = round(linspace(10,M-10,10)); % select training set sizes

if trainFlag
    % isolate training (2/3) and testing (1/3) subsets
    a = find( labels >  0 );
    b = find( labels <= 0 );
    rand('twister', sum(100*clock));
    a_shuffle = randperm(length(a));    %randomize index
    rand('twister', sum(100*clock));
    b_shuffle = randperm(length(b));    %randomize index

    temp = round(3*length(labels)/4); % number of samples in training pool
    temp_a = floor(temp/2);           % number of samples from class "A" in training pool
    temp_b = temp-temp_a;             % number of samples from class "B" in training pool

    training_pool_a = a(a_shuffle(1:temp_a));
    training_pool_b = b(b_shuffle(1:temp_b));
    testing_pool_a = a(a_shuffle(temp_a+1:end));
    testing_pool_b = b(b_shuffle(temp_b+1:end));

    training_pool = [training_pool_a; training_pool_b];
    testing_pool = [testing_pool_a; testing_pool_b];
    fprintf('\nTraining pool:\n'); disp(training_pool);
    fprintf('\nTesting pool:\n'); disp(testing_pool);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

% create T1 random subsets for each of the n training set sizes
if trainFlag
    fprintf('\nGenerating %g subsets each for %g training set sizes...\n',T1,length(n));
    training = cell(T1,length(n)); testing = cell(T1,length(n));
    for i=1:length(n)
        for j=1:T1
            temp_a = training_pool_a(randperm(length(training_pool_a)));
            temp_b = training_pool_b(randperm(length(training_pool_b)));
            temp = interleave(temp_a,temp_b);

            training{j,i} = temp(1:n(i));
            testing{j,i} = testing_pool;
        end
    end
else
    fprintf('\nLoading %g subsets each for %g training set sizes...\n',size(tra_in,1),size(tra_in,2));
    training = tra_in;
    testing = tes_in;
end

%%%% DEBUG %%%%
% save('/data2/home/abasavan/for_scott/experiments/debug_traintest.mat','training','testing','data','labels');
% quit;
%%%%%%%%%%%%%%%%

% run classification for all T1 x n subsets
for j=1:T1
    [junk1,junk2,junk3]=mkdir(num2str(j));
end

err_subsampling = zeros(T1,length(n));
for i=1:length(n)
    fprintf('Running T1=%g classification trials with %g training samples...\n',T1,n(i));
    parfor j=1:T1
        cd(num2str(j));

        curTrainSet = data(training{j,i},:);
        curTestSet = data(testing{j,i},:);
        curTrainLabel = labels(training{j,i});
        curTestLabel = labels(testing{j,i});
                
        stats = Classify('SVM',curTrainSet,curTestSet,curTrainLabel,curTestLabel,'linear');
        
        err_subsampling(j,i) = (stats.fp + stats.fn)/(stats.tp + stats.tn + stats.fp + stats.fn);
        
        cd('..');
    end
end

for j=1:T1
    [junk1 junk2 junk3]=rmdir(num2str(j),'s');
end

% calculate mean, 25th-tile, 75-tile for each n (over all T1 subsets)
err_mean = mean(err_subsampling);
temp = prctile(err_subsampling,[25 75]);
err_25 = temp(1,:);
err_75 = temp(2,:);

%%%% DEBUG %%%%
% save('/data2/home/abasavan/for_scott/experiments/debug_true_errs.mat','err_mean','err_25','err_75','err_subsampling');
% quit;
%%%%%%%%%%%%%%%%

%%% Part B: Permuation Test %%%
% create randomized training labels (T1 x n x T2)
rand_training_labels = cell(T1,length(n),T2);
fprintf('\nGenerating T2=%g randomized subsets for each of T1=%g trials for each of %g training set sizes...\n',T2,T1,length(n));
for i=1:length(n)
    for j=1:T1
        for k=1:T2
%             rand('twister', sum(100*clock));
            temp = randi(2,n(i),1);
            temp = 2*(temp - 1.5);
            rand_training_labels{j,i,k} = temp;
        end
    end
end

% run classification for all T1 x n x T2 subsets
tic
for k=1:T2
    [junk1,junk2,junk3]=mkdir(num2str(k)); % create directories used by parfor
end

err_permutation = zeros(T1,length(n),T2);
for i=1:length(n)
    fprintf('Running T1xT2=%g classification trials with %g training samples...\n',T1*T2,n(i));
    for j=1:T1
        curTrainSet = data(training{j,i},:);
        curTestSet = data(testing{j,i},:);
        curTestLabel = labels(testing{j,i});
        parfor k=1:T2
            cd(num2str(k));
            
            curTrainLabel = rand_training_labels{j,i,k};
            stats = Classify('SVM',curTrainSet,curTestSet,curTrainLabel,curTestLabel,'linear');
            err_permutation(j,i,k) = (stats.fp + stats.fn)/(stats.tp + stats.tn + stats.fp + stats.fn);
            
            cd('..');
        end
    end
end

for k=1:T2
    rmdir(num2str(k),'s'); % delete directories used by parfor
end
toc

%%%% DEBUG %%%%
% save('/data2/home/abasavan/for_scott/experiments/debug_err_permute.mat','err_permutation');
% quit;
%%%%%%%%%%%%%%%%

%%% Part C: Significance Calculation %%%
% Calculate P values for all 3 metrics (i.e. mean, 25th, 75th).
% This is fraction of random classifiers w/ lower error rates than actual.
P_mean = zeros(size(n)); P_25 = zeros(size(n)); P_75 = zeros(size(n));
for i=1:length(n)
    RandErrs = err_permutation(:,i,:);
    
    P_mean(i) = numel(find(RandErrs < err_mean(i)))/numel(RandErrs);
    P_25(i) = numel(find(RandErrs < err_25(i)))/numel(RandErrs);
    P_75(i) = numel(find(RandErrs < err_75(i)))/numel(RandErrs);
end

% for each metric, find smallest n that achieves significance
sig_level = .05;
valid_mean = find(P_mean < sig_level, 1 );
valid_25 = find(P_25 < sig_level, 1 );
valid_75 = find(P_75 < sig_level, 1 );


%%%%%%%%%%%%%%%%%% Learning Curves %%%%%%%%%%%%%%%%%%%
% only select valid n values and their associated error rates
actual.n_mean = n(valid_mean:end);
actual.err_mean = err_mean(valid_mean:end);
actual.n_25 = n(valid_25:end);
actual.err_25 = err_25(valid_25:end);
actual.n_75 = n(valid_75:end);
actual.err_75 = err_75(valid_75:end);

%%%%% debug %%%%%
% save('my_data.mat','actual');
%%%%% /debug %%%%

%%%%% debug %%%%%
% load('my_data.mat');
%%%%% /debug %%%%

% save actual valid error rates and extrapolated curves
extrap.n = 10:400;
extrap.err_mean = InvSqrFit(actual.n_mean, actual.err_mean, extrap.n);
extrap.err_25 = InvSqrFit(actual.n_25, actual.err_25, extrap.n);
extrap.err_75 = InvSqrFit(actual.n_75, actual.err_75, extrap.n);

save('PowerStudy.mat','actual','extrap');
