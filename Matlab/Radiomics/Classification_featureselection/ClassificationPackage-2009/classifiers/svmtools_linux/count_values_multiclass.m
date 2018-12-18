function [ tp, tn, fp, fn] = count_values_multiclass( results , real_values )
tp = 0;
tn = 0;
fp = 0;
fn = 0;

for i = 1:length(results)
    if( results(i) == round(real_values(i)) )
        tp = tp + 1;
    else
        fp = fp + 1;
    end
end

% tpj = 0;
% fpj = 0;
% for k = 1:max(real_values)
%     tp = 0;
%     tn = 0;
%     fp = 0;
%     fn = 0;
%     
%     values = real_values - k;
%     v = find(values == 0);
% 
%     for j = 1:length(results)
%         if( results(j) == round(real_values(j)) )
%             tpj = tpj + 1;
%         else
%             fpj = fpj + 1;
%         end
%     end
%     classaccuracy(k) = tpj/(tpj+fpj);
end
