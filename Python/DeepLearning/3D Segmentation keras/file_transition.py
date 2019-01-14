# coding=utf-8
import os
import shutil

path = '/media/zzr/My Passport/430/CT/Preprocess_CT1'
outpath = '/media/zzr/Data/Task07_Pancreas/ChangHai/img'
a=os.listdir(path)
for idx, i in enumerate(os.listdir(path)):
    print i
    file_ = [j for j in os.listdir(os.path.join(path, i)) if 'P' in j and 'img' in j]
    shutil.copyfile(os.path.join(os.path.join(path, i), file_[0]), os.path.join(outpath, str(idx) + '.nii.gz'))

print 'done'
