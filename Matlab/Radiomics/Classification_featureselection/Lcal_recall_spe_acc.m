function [recall,specificity,accuracy]=Lcal_recall_spe_acc(labels_validation,labels_predict)

recall=sum(labels_predict==1&labels_validation==1)/sum(labels_validation==1)*100;
specificity=sum(labels_predict==0&labels_validation==0)/sum(labels_validation==0)*100;
accuracy=(sum(labels_predict==1&labels_validation==1)+sum(labels_predict==0&labels_validation==0))/length(labels_predict)*100;
fprintf('Accuracy=%.2f, Specificity=%.2f, Precision=%.2f\n',accuracy,specificity,recall);
end
