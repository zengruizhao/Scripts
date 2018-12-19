# coding=utf-8
from xml.dom import minidom
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
from skimage import measure,data,color
import skimage.morphology as sm


def creat_xml(img, resultFile):
    contours = measure.find_contours(img, 0.5)
    doc = minidom.Document()
    ASAP_Annotation = doc.createElement("ASAP_Annotations")
    doc.appendChild(ASAP_Annotation)
    Annotations = doc.createElement("Annotations")
    ASAP_Annotation.appendChild(Annotations)
    for i in xrange(len(contours)):
        Annotation = doc.createElement("Annotation")
        Annotation.setAttribute("Name", "_" + str(i))
        Annotation.setAttribute("Type", "Polygon")
        Annotation.setAttribute("PartOfGroup", "None")
        Annotation.setAttribute("Color", "#F4FA58")
        Annotations.appendChild(Annotation)

        Coordinates = doc.createElement("Coordinates")
        Annotation.appendChild(Coordinates)
        for j in range(len(contours[i])):
            Coordinate1 = doc.createElement("Coordinate")
            Coordinate1.setAttribute("Order", str(j))
            Coordinate1.setAttribute("Y", str(contours[i][j, 0] * 72 + 144))
            Coordinate1.setAttribute("X", str(contours[i][j, 1] * 72 + 144))
            Coordinates.appendChild(Coordinate1)

    f = file(resultFile, "w")
    doc.writexml(f)
    f.close()


if __name__ == "__main__":
    fileName = '/media/zzr/Data/skin_xml/mask_result/2018-07-30 161045.npy'
    npyImg = np.load(fileName)
    img = npyImg == 0
    creat_xml(img,
              resultFile="/run/user/1000/gvfs/afp-volume:host=Darwin-MI.local,user=zrzhao,volume=Data/Ultrasound Nerve Segmentation/temp/0.xml")

