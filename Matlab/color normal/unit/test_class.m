% SVM分类测试
clc;clear all;close all;
addpath('..\cellclass\unit\');
class_1_folder='D:\Program Files\Matlab\work\cellclass\graphfeat5\c\';%15 8+7
class_2_folder='D:\Program Files\Matlab\work\cellclass\SVMtest\c\2\';%16 8+8
class_3_folder='D:\Program Files\Matlab\work\cellclass\SVMtest\c\3\';%18 9+9
num=3;

file_ext = '.mat';
for i=1:400 %多次测试
    fprintf(' 测试 %d 次，结果为:\n ',i);
    % 获取class1训练+测试特征矩阵
    folder_1_content = dir ([class_1_folder,'*',file_ext]);  %文件夹内容
    c1=1:15;
    K=randperm(length(c1));    
    for k=1:10
        name=folder_1_content(K(k),1).name;
        ind=strfind(name,'.mat');
        name=name(1:ind(1)-2);
        for j=1:5
            string = [class_1_folder,name,num2str(j),'.mat'];
            load(string);          
            eval(['c1_feat((k-1)*5+j,:)=vfeature',num2str(j),';']);
        end
    end
    for k=11:15
        name=folder_1_content(K(k),1).name;
        ind=strfind(name,'.mat');
        name=name(1:ind(1)-2);
        for j=1:5
            string = [class_1_folder,name,num2str(j),'.mat'];
            load(string);          
            eval(['c1_test((k-11)*5+j,:)=vfeature',num2str(j),';']);  
        end
    end
    % 获取class2训练+测试特征矩阵
    folder_2_content = dir ([class_2_folder,'*',file_ext]);  %文件夹内容
    c2=1:16;
    K=randperm(length(c2));    
    for k=1:10
        name=folder_2_content(K(k),1).name;
        ind=strfind(name,'.mat');
        name=name(1:ind(1)-2);
        for i=1:5
            string = [class_2_folder,name,num2str(j),'.mat'];
            load(string);          
            eval(['c2_feat((k-1)*5+j,:)=vfeature',num2str(j),';']);
        end
    end
    for k=11:16
        name=folder_2_content(K(k),1).name;
        ind=strfind(name,'.mat');
        name=name(1:ind(1)-2);
        for i=1:5
            string = [class_2_folder,name,num2str(j),'.mat'];
            load(string);          
            eval(['c1_test((k-11)*5+j,:)=vfeature',num2str(j),';']);
        end
    end
    % 获取class3训练+测试特征矩阵
    folder_3_content = dir ([class_3_folder,'*',file_ext]);  %文件夹内容
    c3=1:18;
    K=randperm(length(c3));    
    for k=1:12
        name=folder_3_content(K(k),1).name;
        ind=strfind(name,'.mat');
        name=name(1:ind(1)-2);
        for j=1:5
            string = [class_3_folder,name,num2str(j),'.mat'];
            load(string);          
            eval(['c3_feat((k-1)*5+j,:)=vfeature',num2str(j),';']);
        end
    end
    for k=13:18
        name=folder_3_content(K(k),1).name;
        ind=strfind(name,'.mat');
        name=name(1:ind(1)-2);
        for j=1:5
            string = [class_3_folder,name,num2str(j),'.mat'];
            load(string);          
            eval(['c3_test((k-13)*5+j,:)=vfeature',num2str(j),';']);  
        end
    end
    class_train=[c1_feat;c2_feat;c3_feat];
    train_label=[ones(10,1);2*ones(10,1);3*ones(12,1)];    
    class_test=[c1_test;c2_test;c3_test];
    test_label=[ones(5,1);2*ones(6,1);3*ones(6,1)];    
    matrix1=[class_train; class_test];
    matrix1=[matrix1(:,18),matrix1(:,22),matrix1(:,24),matrix1(:,26),matrix1(:,35),matrix1(:,36),matrix1(:,46)];%c
%     matrix1=[matrix1(:,17),matrix1(:,32),matrix1(:,33),matrix1(:,43),matrix1(:,18),matrix1(:,22),matrix1(:,38)];%l

    matrix=mapminmax(matrix1',0,1)';
    [pc,score,latent,tsquare]= princomp(matrix);
    matrix=score(:,1:num);

    trainmatrix=matrix(1:32,:);
    testmatrix=matrix(33:49,:);
    model= svmtrain(train_label,trainmatrix);
    [CR tongji]= ClassResult(test_label,testmatrix,model,2);
    preaccu(:,i)=[tongji([1 2],1);tongji([1 2],2);tongji(:,3)]; 
    [predict,accuracy,pre]=svmpredict(test_label,testmatrix,model);
    preclass(:,i)=predict;
end
fprintf('\n\n\n 随机测试 %d 次，结果为：\n ',i);
accury(1)=mean(preaccu(2,:),2);
accury(5)=max(preaccu(2,:));
fprintf('class1整体准确率为 %3.2f / %3.2f\n ',accury(1),accury(5));
accury(2)=mean(preaccu(4,:),2);
accury(6)=max(preaccu(4,:));
fprintf('class2整体准确率为 %3.2f / %3.2f\n ',accury(2),accury(6));
accury(3)=mean(preaccu(6,:),2);
accury(7)=max(preaccu(6,:));
fprintf('class3整体准确率为 %3.2f / %3.2f\n ',accury(3),accury(7));
accury(4)=mean(preaccu(7,:));
accury(8)=max(preaccu(7,:));
fprintf('整体准确率为 %3.2f / %3.2f\n ',accury(4),accury(8));

    % PCA降维    
%     [pc,score,latent,tsquare]= princomp(trainmatrix); 
%     trainmatrix=score(:,1:num);
%     [pc,score,latent,tsquare]= princomp(testmatrix); 
%     testmatrix=score(:,1:num);
%     conpercent = cumsum(latent)./sum(latent)   

