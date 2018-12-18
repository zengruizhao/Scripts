# coding=utf-8
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops, label
from get_small_patch import hole_fill
from xml.dom import minidom
from collections import OrderedDict


color = OrderedDict([("r", "#ff0000"), ("g", "#00ff00"), ("b", "#0000ff")])
Channel = OrderedDict([('0', 'r'), ('1', 'g'), ('2', 'b')])


def process(mask):
    mask = label(hole_fill(remove_small_objects(mask.astype(np.bool), 100)))
    mask = regionprops(mask)
    mask_coord = [i.centroid for i in mask]
    return mask_coord


path = '/run/user/1000/gvfs/afp-volume:host=Darwin-MI.local,user=zrzhao,volume=Data/Ultrasound Nerve Segmentation/temp/result'
doc = minidom.Document()
ASAP_Annotation = doc.createElement("ASAP_Annotations")
doc.appendChild(ASAP_Annotation)
Annotations = doc.createElement("Annotations")
ASAP_Annotation.appendChild(Annotations)

for ii, name in enumerate([i for i in os.listdir(path) if i.endswith('.png')]):
    print ii, '/', len(os.listdir(path))
    row = int(name.split('.')[0].split('_')[1])-1
    column = int(name.split('.')[0].split('_')[-1])-1
    mask = cv2.imread(os.path.join(path, name))
    red, green, blue = mask[..., 0], mask[..., 1], mask[..., 2]
    red_coord = process(red)
    green_coord = process(green)
    blue_coord = process(blue)

    # fig, ax = plt.subplots(2, 2)
    # ax[0, 0].imshow(mask)
    # ax[0, 1].imshow(red)
    # ax[1, 0].imshow(green)
    # ax[1, 1].imshow(blue)
    # plt.show()
    for channel, coord in enumerate([red_coord, green_coord, blue_coord]):
        color_channel = Channel[str(channel)]
        for idx, j in enumerate(coord):
            Annotation = doc.createElement("Annotation")
            Annotation.setAttribute("Name", "Annotation " + str(idx))
            Annotation.setAttribute("Type", "Dot")
            Annotation.setAttribute("PartOfGroup", "None")
            Annotation.setAttribute("Color", color[color_channel])
            Annotations.appendChild(Annotation)
            Coordinates = doc.createElement("Coordinates")
            Annotation.appendChild(Coordinates)
            ##############################
            y = j[0]+1024*row
            x = j[1]+1024*column
            Coordinate1 = doc.createElement("Coordinate")
            Coordinate1.setAttribute("Order", '0')
            Coordinate1.setAttribute("Y", str(y))
            Coordinate1.setAttribute("X", str(x))
            Coordinates.appendChild(Coordinate1)

f = file('/run/user/1000/gvfs/afp-volume:host=Darwin-MI.local,user=zrzhao,volume=Data/Ultrasound Nerve Segmentation/temp/dot.xml', "w")
doc.writexml(f)
f.close()
