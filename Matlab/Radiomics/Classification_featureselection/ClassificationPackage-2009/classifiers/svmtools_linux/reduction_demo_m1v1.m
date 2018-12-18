function reduction_demo
warning off
datasets = [17]; %choose datasets
dimensions = [2:10]; %choose dimensions
methods = [5 6 9 1 2 3]; %choose methods
trunk = 2;
fs = 1; %feature selection? fs = 0; %off fs = 1; %on
if fs == 0
    %load svm_stats_10_31_07
else
    %load svm_stats_7_19_07
end
%% Default krange = [1] runs Isomaps and LLE for all possible k values 2 through n-1,
%% where n = the total number of points in the dataset

%% You can set a smaller range of k, by changing the krange parameter
%trunk = [2]; %run for what range of k values for Isomap/LLE

%load datasets
%1 = Prostatetumortest
%2 = Prostatetumortrain
%3 = Colontumor
%4 = Ovariantumor
% %5 = Prostaterelapse
% %6 = Breastrelapsetest
% %7 = Breastrelapsetrain
%8 = Lungcancertest
%9 = Lungcancertrain
%10 = AML vs ALL Leukemia test
%11 = AML vs ALL Leukemia train
%12 = diffuse large B-cell lymphoma (DLBCL)
%13 = Central Nervous System
%14 = Lungcancermichigan
%15 = diffuse large B-cell lymphoma (DLBCL) vs Follicular lymphoma (FL)
%16 = diffuse large B-cell lymphoma (DLBCL) outcome
%18 = Grade 3 vs Grade 4 Glioma
%21 = Breast Cancer

%%--------------------------begin dataset--------------------------------%%
for dataset = datasets
    [X, labels, titlestring] = loadthis(dataset); %open dataset
    labelindices = find(labels < 6);
    X = X(labelindices,:);
    labels = labels(labelindices)';
    diary off %turn off diary

    %%---------------------begin feature selection---------------------------%%
    if fs ==1, X = selectfeatures(X,dataset); fprintf('feature selection on '); size(X), end
    %%---------------------end feature selection-----------------------------%%

    krange = [2:(length(labels)-1)]; %run for 'full' range of k values for Isomap/LLE
    krange = trunk; %run for smaller range of k values for Isomap/LLE
    fprintf('%s\n', titlestring);
    fprintf('Negatives: %i , Positives: %i , ratio: %2.10f \n ', length( find(labels<=0) ), length( find(labels>0) ), length( find(labels>0 ))/length(labels) );
    %%-------------------------begin dimensions------------------------------%%
    for newdim = dimensions
        dims = [1:newdim];
        fprintf(['Dimensions 1 through ' num2str(newdim) '\n']);
        %%-------------------------begin methods ---------------------------------%%
        for method = methods
            %%------------------------begin k embeddings-----------------------------%%
            if method ==5 | method ==6 | method == 8| method == 9
                %% run for several k values
                for k = krange
                    t_start = clock;
                    [temp, methodstring] = Dimred(method, X, newdim, labels,k);  %% Run Dim Reduction
                    if size(temp,1) ~= size(X,1), continue, end %% if outliers removed, run with next k
                    t_stop = clock;
                    time = etime(t_stop,t_start);  %% Calculate Run Time

                    if dataset == 17
                        [tp, tn, fp, fn] = run_demo_multiclass1v1(temp, labels);
                    else
                        [tp, tn, fp, fn] = run_demo(temp, labels);
                    end
                    acc = (tp+tn)/(tp+tn+fp+fn);
                    spec = (tn)/(tn+fp);
                    sens = (tp)/(tp+fn);

                    acc_vec(1,k) =  acc;
                    sens_vec(1,k) =  sens;
                    spec_vec(1,k) =  spec;
                    time_vec(1,k) =  time;
                    k
                end

                %% Calcuate first 4 metrics Acc/Spec/Sens/Time
                [acc,bestk] = max(acc_vec);
                sens = sens_vec(bestk);
                spec = spec_vec(bestk);
                time = time_vec(bestk);

                %% Record results for each k
                allkacc(dataset,newdim,method,1:length(acc_vec)) = acc_vec;
                allkspec(dataset,newdim,method,1:length(acc_vec)) = sens_vec;
                allksens(dataset,newdim,method,1:length(acc_vec)) = spec_vec;
                allktime(dataset,newdim,method,1:length(acc_vec)) = time_vec;

                %% Average Results for Accuracy/Spec/Sens/Time
                avg_acc_table(dataset, newdim, method) = sum(acc_vec(krange))/length(krange); %% Accuracy Table
                avg_spec_table(dataset, newdim, method) = sum(spec_vec(krange))/length(krange); %% Specificity Table
                avg_sens_table(dataset, newdim, method) = sum(sens_vec(krange))/length(krange); %% Sensitivity Table
                avg_time_table(dataset, newdim, method) = sum(time_vec(krange))/length(krange); %% Run Time Table

                fprintf('%f %f %f %f %s, k = %s\n', acc, sens, spec, time, methodstring, num2str(bestk));
                fprintf('%f %f %f %f\n',avg_acc_table(dataset, newdim, method),avg_sens_table(dataset, newdim, method),avg_spec_table(dataset, newdim, method),avg_time_table(dataset, newdim, method))
                fprintf('%f %f\n', std(acc_vec(krange)), std(time_vec(krange)))
                %%------------------------end k embeddings-----------------------------%%
            else
                %%-------------------------begin other embeddings------------------------%%
                t_start = clock;
                [temp, methodstring] = Dimred(method, X, newdim, labels,1); %% Run Dim Reduction
                t_stop = clock;

                time = etime(t_stop,t_start);  %% Calculate Run Time
                if dataset == 17
                    [tp, tn, fp, fn] = run_demo_multiclass1v1(temp, labels);
                else
                    [tp, tn, fp, fn] = run_demo(temp, labels);
                end

                acc = (tp+tn)/(tp+tn+fp+fn);
                spec = (tn)/(tn+fp);
                sens = (tp)/(tp+fn);
                fprintf('%f %f %f %f %s\n', acc, sens, spec, time, methodstring);
                %%-------------------------end other embeddings------------------------%%
            end

            %%---------------------- Begin Calculating Metrics ----------------------%%
            %%Final Results for Accuracy/Spec/Sens/Time
            acc_table(dataset, newdim, method) = acc;   %% Accuracy Table
            spec_table(dataset, newdim, method) = spec; %% Specificity Table
            sens_table(dataset, newdim, method) = sens; %% Sensitivity Table
            time_table(dataset, newdim, method) = time; %% Run Time Table

            if method == 5 |method ==6 |method ==8| method == 9
                k_table(dataset,newdim,method) = bestk;  %% Parameter 'K' Table
                precision_acc_table(dataset,newdim,method) = std(acc_vec(krange));  %% Standard Deviation of Accuracy
                precision_time_table(dataset,newdim,method) = std(time_vec(krange)); %% Standard Deviation of Run Time
            end
            %%---------------------- end Calculating Metrics ----------------------%%
            %%-------------------------end methods------------------------------%%
        end
        %%-------------------------end dimensions------------------------------%%
    end
    if fs == 0
        save('svm_stats_10_31_07.mat','acc_table','spec_table','sens_table','time_table','allkacc','allksens','allkspec','allktime','avg_acc_table','avg_spec_table','avg_sens_table','avg_time_table','k_table','precision_acc_table','precision_time_table');fprintf('Saved.');
    else
        save('svm_stats_10_31_07.mat','acc_table','spec_table','sens_table','time_table','allkacc','allksens','allkspec','allktime','avg_acc_table','avg_spec_table','avg_sens_table','avg_time_table','k_table','precision_acc_table','precision_time_table');fprintf('Saved.');
    end
    %%--------------------------end dataset--------------------------------%%
end

diary off
