function set_candiF=Lpick_top_n_features_with_pvalue_correlation(data_ccm,idx_TTest,num_features,correlation_factor)
if nargin<4
    correlation_factor=0.6;
end
X=data_ccm(:,idx_TTest);
[RHO]=corr(X);
set_candiF(1)=idx_TTest(1);
idx_pool=idx_TTest;
idx_pool(abs(RHO(1,:))>correlation_factor)=[];

for i=2:num_features
    if numel(idx_pool)>0
        set_candiF(i)=idx_pool(1);
        X=data_ccm(:,idx_pool);
        [RHO]=corr(X);
        idx_pool(abs(RHO(1,:))>correlation_factor)=[]; %feature_list_ccm(set_candiF(i))
    else
        break;
    end
end
