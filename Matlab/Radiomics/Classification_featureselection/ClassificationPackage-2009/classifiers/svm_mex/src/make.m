function make(varargin)
% A Matlab-based Makefile.
% add -largeArrayDims on 64-bit machines

% parse inputs
switch nargin
    case 0
        opt='all';
    case 1
        opt=varargin{1};
    otherwise
        error('No more than 1 input allowed.');
end

% set source and binary paths
src_dir = pwd;
bin_dir = ['..' filesep 'bin'];

switch lower(opt)
    case {'all'}
        make_all(src_dir,bin_dir);
    case {'clean'}
        make_clean(src_dir,bin_dir);
    otherwise
        error('Unrecognized option.')
end


function make_all(src_dir,bin_dir)
% find computer architecture
% currently supports 32- and 64-bit Windows and *nix systems
C = computer;
OS = C(1:2);
arch = C(end-1:end);
ext = mexext;

if strcmp(OS,'PC') && ~strcmp(C(end-1:end),'64') % 32-bit windows
    mex -O -c svm.cpp
    mex -O -c svm_model_matlab.cpp
    mex -O svmtrain.cpp svm.obj svm_model_matlab.obj
    mex -O svmpredict.cpp svm.obj svm_model_matlab.obj
%     mex -O libsvmread.cpp
%     mex -O libsvmwrite.cpp
elseif strcmp(OS,'PC') && strcmp(C(end-1:end),'64') % 64-bit windows
    mex -O -c -largeArrayDims svm.cpp
    mex -O -c -largeArrayDims svm_model_matlab.cpp
    mex -O -largeArrayDims svmtrain.cpp svm.obj svm_model_matlab.obj
    mex -O -largeArrayDims svmpredict.cpp svm.obj svm_model_matlab.obj
%     mex -O -largeArrayDims libsvmread.cpp
%     mex -O -largeArrayDims libsvmwrite.cpp
elseif strcmp(OS,'GL') && strcmp(C(end-1:end),'64') % 64-bit linux
    mex -O -c -largeArrayDims svm.cpp
    mex -O -c -largeArrayDims svm_model_matlab.cpp
    mex -O -largeArrayDims svmtrain.cpp svm.o svm_model_matlab.o
    mex -O -largeArrayDims svmpredict.cpp svm.o svm_model_matlab.o
%     mex -O -largeArrayDims libsvmread.cpp
%     mex -O -largeArrayDims libsvmwrite.cpp
elseif strcmp(OS,'GL') && ~strcmp(C(end-1:end),'64') % 32-bit linux
    mex -O -c -L/usr/lib/gcc/x86_64-linux-gnu/4.4.4/32 svm.cpp
    mex -O -c -L/usr/lib/gcc/x86_64-linux-gnu/4.4.4/32 svm_model_matlab.cpp
    mex -O -L/usr/lib/gcc/x86_64-linux-gnu/4.4.4/32 svmtrain.cpp svm.o svm_model_matlab.o
    mex -O -L/usr/lib/gcc/x86_64-linux-gnu/4.4.4/32 svmpredict.cpp svm.o svm_model_matlab.o
%     mex -O -L/usr/lib/gcc/x86_64-linux-gnu/4.4.4/32 libsvmread.cpp
%     mex -O -L/usr/lib/gcc/x86_64-linux-gnu/4.4.4/32 libsvmwrite.cpp
end

% move compiled mex files into bin directory
% names are changed to avoid conflicts with matlab's built-in SVM functions
movefile([src_dir filesep 'svmtrain.' ext],[bin_dir filesep 'trainsvm.' ext],'f');
movefile([src_dir filesep 'svmpredict.' ext],[bin_dir filesep 'predictsvm.' ext],'f');


function make_clean(src_dir,bin_dir)
s = warning('off', 'all');
ext = mexext;
delete([src_dir filesep '*.obj']);
delete([src_dir filesep '*.o']);
delete([bin_dir filesep 'trainsvm.' ext]);
delete([bin_dir filesep 'predictsvm.' ext]);
warning(s);

