clear all;
clc;
dataPath = '/media/zzr/Lens/data/train_test/';
path = dir(dataPath);
for i =3:numel(path)
    if i == 3
            fp = fopen([dataPath,'testList.txt'],'wt');
    else
            fp = fopen([dataPath,'trainList.txt'],'wt');
    end
      subPath = dir([dataPath,path(i).name]);
      for j = 3:numel(subPath)
       	img = dir([dataPath,path(i).name,'/',subPath(j).name,'/*.png']);%%
        for ii = 1:numel(img)
              fprintf(fp,'%s\n',[subPath(j).name,'/',img(ii).name,' ',num2str(j-3)]);
        end
      end
      fclose(fp);
end
