# coding=utf-8
from __future__ import print_function
import os
import SimpleITK as sitk
import six
import matplotlib.pyplot as plt
from radiomics import featureextractor, setVerbosity, getTestCase
resultPath = '/home/zzr/git/pyradiomics1111/result'

dataDir = '/home/zzr/git/pyradiomics1111'
imageName, maskName = getTestCase('brain1', dataDir)
# imageName = os.path.join('/media/zzr/My Passport/430/MRI/Preprocess_MRI/03534455_WANG YONG SHENG_MR', 'T2_img.nii.gz')
# maskName = os.path.join('/media/zzr/My Passport/430/MRI/Preprocess_MRI/03534455_WANG YONG SHENG_MR', 'T2_label.nii.gz')
params = os.path.join(dataDir, 'examples', 'exampleSettings', 'mine.yaml')
extractor = featureextractor.RadiomicsFeaturesExtractor(params)
extractor.addProvenance(provenance_on=False)
extractor.enableAllFeatures()
# extractor.enableFeatureClassByName('shape', False)
extractor.enableAllImageTypes()
setVerbosity(60)
# image, mask = extractor.loadImage(imageName, maskName)
# waveletImage = list(imageoperations.getWaveletImage(image, mask))
# print(len(waveletImage))
# img = sitk.GetArrayFromImage(waveletImage(0))
# for i in xrange(img.shape[0]):
#     plt.imshow(img[i, ...], cmap=plt.cm.gray)
#     plt.show()
# print(waveletImage)
# for x in waveletImage:
#     img = sitk.GetArrayFromImage(x[0])
#     print(img.shape)
#     for i in xrange(img.shape[0]):
#         plt.imshow(img[i, ...], cmap=plt.cm.gray)
#         plt.show()
#     break
result = extractor.execute(imageName, maskName, label=1)
for key, val in six.iteritems(result):
    print(key, val)

print(len(result))
# result = extractor.execute(imageName, maskName, voxelBased=True)
# for key, val in six.iteritems(result):
#     if isinstance(val, sitk.Image):
#         sitk.WriteImage(val, os.path.join(resultPath, key+'.nrrd'))
#         print('Stored feature %s in %s' % (key, key + '.nrrd'))
#     else:
#         print('\t%s: %s' % (key, val))

