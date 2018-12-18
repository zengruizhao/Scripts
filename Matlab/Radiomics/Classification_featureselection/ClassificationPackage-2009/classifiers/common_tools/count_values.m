function [ tp , tn , fp , fn ] = count_values( results , real_values ) 
tp = 0;
tn = 0;
fp = 0;
fn = 0;

for i = 1:length(results)
    if( sign( results(i) ) == real_values(i) )
        if( sign(results(i)) > 0 )
            tp = tp + 1;
        else
            tn = tn + 1;
        end
    else
        if( sign(results(i)) > 0 )
            fp = fp + 1;
        else
            fn = fn + 1;
        end            
    end
end