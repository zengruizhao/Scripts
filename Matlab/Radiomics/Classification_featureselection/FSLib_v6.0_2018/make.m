% MAKE ALL
%  Matlab Code-Library for Feature Selection
%  A collection of S-o-A feature selection methods
%  Version 6.0 March 2018
%  Support: Giorgio Roffo
%  E-mail: giorgio.roffo@glasgow.ac.uk

disp('+ Build and install');
addpath(genpath(['.',filesep,'compiletools']));
cd('lib');

list = dir('*.cpp');
for i=1:length(list),
    fprintf('building mex(dll) of %s \n',list(i).name);
    if strcmp(list(i).name,'libsvm_classifier_spider.cpp')==0
        if strcmp(list(i).name,'svm.cpp')==0
            mex([list(i).name]);
        end
    else
        eval(['mex libsvm_classifier_spider.cpp svm.cpp -Icompiletools ','compiletools',filesep,'mexarg.cpp']) 
    end
end

cd('drtoolbox')
mexall


cd('..');
cd('..');



disp('+ Completed!');

