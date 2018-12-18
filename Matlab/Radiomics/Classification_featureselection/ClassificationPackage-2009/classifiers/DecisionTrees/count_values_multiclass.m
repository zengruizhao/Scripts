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