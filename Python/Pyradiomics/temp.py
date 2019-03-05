# coding=utf-8
from __future__ import print_function
import os
import SimpleITK as sitk
import six
import matplotlib.pyplot as plt
from radiomics import featureextractor, getTestCase, imageoperations, generalinfo
resultPath = '/home/zzr/git/pyradiomics1111/result'

dataDir = '/home/zzr/git/pyradiomics1111'
imageName, maskName = getTestCase('brain1', dataDir)
params = os.path.join(dataDir, 'examples', 'exampleSettings', 'Params.yaml')
extractor = featureextractor.RadiomicsFeaturesExtractor(params)
extractor.addProvenance(provenance_on=False)
# extractor.enableAllFeatures()
extractor.enableAllImageTypes()
# image, mask = extractor.loadImage(imageName, maskName)
# waveletImage = list(imageoperations.getLBP2DImage(image, mask))
# print(list(waveletImage))
# img = sitk.GetArrayFromImage(waveletImage)
# for i in xrange(img.shape[0]):
#     plt.imshow(img[i, ...], cmap=plt.cm.gray)
#     plt.show()
# print(waveletImage)
# for x in waveletImage:
#     img = sitk.GetArrayFromImage(x[0])
#     for i in xrange(img.shape[0]):
#         plt.imshow(img[i, ...], cmap=plt.cm.gray)
#         plt.show()
#     break
result = extractor.execute(imageName, maskName)
print(len(result))
for key, val in six.iteritems(result):
    print(key, val)

# result = extractor.execute(imageName, maskName, voxelBased=True)
# for key, val in six.iteritems(result):
#     if isinstance(val, sitk.Image):
#         sitk.WriteImage(val, os.path.join(resultPath, key+'.nrrd'))
#         print('Stored feature %s in %s' % (key, key + '.nrrd'))
#     else:
#         print('\t%s: %s' % (key, val))

