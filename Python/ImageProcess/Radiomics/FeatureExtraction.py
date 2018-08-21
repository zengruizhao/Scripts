# coding=utf-8
from radiomics import featureextractor, shape, firstorder, glcm, gldm, glrlm, glszm, ngtdm
import sys, os
import SimpleITK as sitk
from xlwt import *

tumor_path = '../tumor'
mask_path = '../mask'
Tumor = os.listdir(tumor_path)
Mask = os.listdir(mask_path)
file = Workbook(encoding='ascii')
table = file.add_sheet('data')
for idx in xrange(len(Tumor)):
    image = sitk.ReadImage(os.path.join(tumor_path, Tumor[idx]))
    mask = sitk.ReadImage(os.path.join(mask_path, Mask[idx]))

    shapeFeatures = shape.RadiomicsShape(image, mask)
    firstorderFeatures = firstorder.RadiomicsFirstOrder(image, mask)
    glcmFeatures = glcm.RadiomicsGLCM(image, mask)
    gldmFeatures = gldm.RadiomicsGLDM(image, mask)
    glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask)
    glszmFeatures = glszm.RadiomicsGLSZM(image, mask)
    ngtdmFeatures = ngtdm.RadiomicsNGTDM(image, mask)

    shapeFeatures.enableAllFeatures()
    glcmFeatures.enableAllFeatures()
    firstorderFeatures.enableAllFeatures()
    gldmFeatures.enableAllFeatures()
    glrlmFeatures.enableAllFeatures()
    glszmFeatures.enableAllFeatures()
    ngtdmFeatures.enableAllFeatures()

    shapeFeatures.execute()
    glszmFeatures.execute()
    glcmFeatures.execute()
    firstorderFeatures.execute()
    gldmFeatures.execute()
    glrlmFeatures.execute()
    ngtdmFeatures.execute()

    AllFeatures = dict({'file_name': Tumor[idx]}.items() +
                       shapeFeatures.featureValues.items() +
                       glszmFeatures.featureValues.items() +
                       glcmFeatures.featureValues.items() +
                       firstorderFeatures.featureValues.items() +
                       gldmFeatures.featureValues.items() +
                       glrlmFeatures.featureValues.items() +
                       ngtdmFeatures.featureValues.items())
    temp = 0
    for key, value in AllFeatures.items():
        # print key, value
        if idx == 0:
            table.write(idx, temp, key)
            table.write(idx+1, temp, value)
        else:
            table.write(idx+1, temp, value)
        temp += 1

    file.save('result.xls')

    # with open('result.csv', 'w') as file:
    #     writer = csv.writer(file)
    #     for key in AllFeatures:
    #         writer.writerow([key, AllFeatures[key]])


