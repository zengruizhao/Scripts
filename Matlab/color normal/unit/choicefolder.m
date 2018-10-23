function choicefolder(string,featchoice,features)
eval([string '=features;']);
str='..\cellclass\mat2\';
switch featchoice
    case 1
        folder='hdgs\';
        save([str,folder,string],string);
    case 2
        folder='statxture\';
        save([str,folder,string],string);
    case 3
%         folder='colorMoments+++\';
        folder='featbyleid2c2\';
        save([str,folder,string],string); 
    case 4
        folder='featbylei\';
        save([str,folder,string],string); 
    case 5
        folder='hog\';
        save([str,folder,string],string); 
end