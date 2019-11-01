# coding=utf-8
from __future__ import print_function
import os
import six
from radiomics import featureextractor, setVerbosity
import SimpleITK as sitk
import scipy.io as mat
import numpy as np
case_2cm = mat.loadmat('/media/zzr/My Passport/430/MRI/tumor_2cm.mat')['tumor_2cm'][:, 0]
grade = mat.loadmat('/media/zzr/My Passport/430/MRI/tumor_2cm.mat')['tumor_2cm'][:, 1]
yamlDir = '/home/zzr/git/pyradiomics1111'
params = os.path.join(yamlDir, 'examples', 'exampleSettings', 'mine_featuremap.yaml')
extractor = featureextractor.RadiomicsFeaturesExtractor(params)
extractor.addProvenance(provenance_on=False)
# extractor.enableAllFeatures()
# extractor.enableFeatureClassByName('shape', False)
# extractor.enableImageTypeByName('lbp', False)
# extractor.enableAllImageTypes()
# extractor.disableAllImageTypes()
# extractor.enableImageTypeByName('LoG', True)
setVerbosity(60)
# file = Workbook()
# table = file.create_sheet('data')
dataDir = '/media/zzr/My Passport/430/MRI/IntensityStandardization_nii'
outPathBase = '/media/zzr/Data/Scripts/Python/Pyradiomics/featureMap/T2'
for case in sorted(os.listdir(dataDir)):
    if case in case_2cm:
        G = grade[np.where(case_2cm == case)][0][0][0]
        if G > 1:
            outPath = os.path.join(outPathBase, '2')
        else:
            outPath = os.path.join(outPathBase, '1')
        print(case)
        if not os.path.exists(os.path.join(outPath, case)):
            os.makedirs(os.path.join(outPath, case))
        casePath = os.path.join(dataDir, case)
        img = [i for i in os.listdir(casePath) if 'img' in i and 'T2' in i]
        label = [i for i in os.listdir(casePath) if 'label' in i and 'T2' in i]
        for idxx, i in enumerate(img):
            imageName = os.path.join(casePath, i)
            maskName = os.path.join(casePath, label[idxx])
            result = extractor.execute(imageName, maskName, label=1, voxelBased=True)
            for key, val in six.iteritems(result):
                sitk.WriteImage(val, os.path.join(os.path.join(outPath, case), key + i))
                # nib.save(val, os.path.join(os.path.join(outPath, case), key))
