%used to create latex ready table by combining  mean and std values
% the tbl should have the same dimension
function table_final=LcreateLatex_Table_with_mean_std(tbl_mean,tbl_std)
table_final=[];

for i=1:size(tbl_mean,1)
    for j=1:size(tbl_std,2)
        table_final{i,j}=sprintf('%.2f$\\pm$%.2f',tbl_mean(i,j),tbl_std(i,j));
    end
end
end
