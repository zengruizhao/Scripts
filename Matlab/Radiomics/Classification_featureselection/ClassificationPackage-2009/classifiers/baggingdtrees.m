function [final,labels,error] = baggingdtrees(tr_vals,tr_labs,tst_vals,tst_labs)%,run)
% [final,labels,error] = baggingdtrees(tr_vals,tr_labs,tst_vals,tst_labs,run)
% Uses some FRAC of training data to create decision tree
% Runs bagging using N decision trees
% Iteratively calls RUN_DECISION_TREE using folder
% ['_valstemp_',num2str(run)]
% Satish Viswanath, Jan 2009
frac=1/4;
N=25;
list = samplerand(N,0,1000000);

% 1 is cancer, 2 is benign

run=1;

%Bagging
baggsetsize = fix(size(tr_vals,1)*(frac));
% fprintf('\nSet %d, Bagging: ',run);
for i=1:N
    newinds1=randintrlv(1:size(tr_vals,1),list(i));
    newinds1=newinds1(1:baggsetsize);

    labels(:,i) = run_decision_tree_satish('_valstemp_',tr_vals(newinds1,:),tr_labs(newinds1),tst_vals,tst_labs);%run);                
    error(i) = length(find(tst_labs~=labels(:,i)))/length(labels);
%     if ~(mod(i,5))
%         fprintf('%d%%..',(i/5)*10); 
%     elseif mod(i,5)>2 
%        fprintf('.');
%     end
    labels(find(labels(:,i)==-1),i)=2;labels(find(labels(:,i)==-2),i)=1;
end

%Generating probablity based off voting bagged d-trees
div = round(size(labels,1)/10);tic;
% fprintf('\nSet %d, Finding final classification: ',run);
for i=1:size(labels,1)
    final(i) = length(find(labels(i,:)==1))/size(labels,2);
%     if mod(i,div)<0, % print a dot for every row. change number to skip rows
%         dt=toc;
%         if dt<4,
%             fprintf('-');
%         else
%             fprintf('.');pause(0.04);    % to facilitate a break
%         end
%        tic
%    end
%    milestones=round(linspace(1,size(labels,1),11));
%    milestone_percents=0:10:100;
%    if any(milestones==i),
%        fprintf('%d%%..',milestone_percents(milestones==i));pause(0.04);
%    end
end

end