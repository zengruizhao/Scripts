%% this function didn't do the feature ranking/sorting, the sorting of important features should be done outside the function, and provided the idx via 
% idx_ranked_feats_descend
function set_candiF=Lpick_top_n_features_with_pvalue_correlation(data_ccm,idx_ranked_feats_descend,num_features,correlation_factor)
if nargin<4
    correlation_factor=0.6;
end
X=data_ccm(:,idx_ranked_feats_descend);
[RHO]=corr(X);
set_candiF(1)=idx_ranked_feats_descend(1);
idx_pool=idx_ranked_feats_descend;
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
