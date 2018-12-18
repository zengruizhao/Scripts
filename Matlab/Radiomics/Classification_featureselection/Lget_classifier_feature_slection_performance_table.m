function T=Lget_classifier_feature_slection_performance_table(resultACC,resultAUC,para)
% below are the script that can generate a table and I can copy/paste in the latex table
set_classifier=para.set_classifier;
set_featureselectioon=para.set_featureselection;

word_tb=zeros(length(para.set_classifier)*length(para.set_featureselection),4);
tbl_classifier_featselection_comb=[];
for i_c=1:length(set_classifier)
    for i_f=1:length(set_featureselectioon)
        tbl_classifier_featselection_comb{(i_c-1)*length(set_featureselectioon)+i_f}=[set_classifier{i_c} '+' set_featureselectioon{i_f}];
        for i_m=1:5%for the metric
            if i_m==1
                word_tb((i_c-1)*length(set_featureselectioon)+i_f,i_m)=resultAUC(i_c,i_f).mean;
                word_tb_std((i_c-1)*length(set_featureselectioon)+i_f,i_m)=1.96*resultAUC(i_c,i_f).std;
            end
            if i_m==2
                word_tb((i_c-1)*length(set_featureselectioon)+i_f,i_m)=resultACC(i_c,i_f).mean_acc;
                word_tb_std((i_c-1)*length(set_featureselectioon)+i_f,i_m)=1.96*resultACC(i_c,i_f).std_acc;
            end
            if i_m==3
                word_tb((i_c-1)*length(set_featureselectioon)+i_f,i_m)=resultACC(i_c,i_f).mean_spec;
                word_tb_std((i_c-1)*length(set_featureselectioon)+i_f,i_m)=1.96*resultACC(i_c,i_f).std_spec;                
            end
            if i_m==4
                word_tb((i_c-1)*length(set_featureselectioon)+i_f,i_m)=resultACC(i_c,i_f).mean_sens;
                word_tb_std((i_c-1)*length(set_featureselectioon)+i_f,i_m)=1.96*resultACC(i_c,i_f).std_sens;
            end
            if i_m==5
                word_tb((i_c-1)*length(set_featureselectioon)+i_f,i_m)=resultACC(i_c,i_f).mean_kappa;
                word_tb_std((i_c-1)*length(set_featureselectioon)+i_f,i_m)=1.96*resultACC(i_c,i_f).std_kappa;  
            end   
        end
    end
end
tbl_classifier_featselection_comb=tbl_classifier_featselection_comb';

tbl_final=LcreateLatex_Table_with_mean_std(word_tb,word_tb_std);

T = array2table(tbl_final,...
    'VariableNames',{'AUC' 'Accuracy' 'Specificity' 'Sensitivity' 'Kappa'});
T=[tbl_classifier_featselection_comb T];
