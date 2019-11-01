clear;clc;
path = '/media/zzr/My Passport/430/MRI/Preprocess_MRI/';
case_file = dir(path);
load '/media/zzr/My Passport/430/MRI/MRI_PNET_raw.mat'
idx = 1;
for i=3:length(case_file)
    phase = dir([path case_file(i).name '/*.nii.gz']);
    for jj=1:size(phase, 1)
        if contains(phase(jj).name(1:end-4), 'T2') && contains(phase(jj).name(1:end-4), 'img')
            for j=2:size(raw, 1)
               ID = raw{j, 4};
               if ischar(ID)==0;ID=num2str(ID);end
               tumor_size = raw{j, 12};       
               if ischar(tumor_size);tumor_size = str2double(tumor_size); end
               if ((~isempty(strfind(case_file(i).name, ID))) && (tumor_size<=20))
                   cell{idx, 1} = case_file(i).name;
                   cell(idx, 2) = raw(j, 7);
                   idx = idx + 1;
                   break;
               end
            end
        end
    end
end