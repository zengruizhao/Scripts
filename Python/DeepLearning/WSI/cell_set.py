# coding=utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pointxml import process


def main():
    cell_size = 32 / 2
    outpath = '/media/zzr/Data/skin_xml/cell/data/patch'
    path = '/media/zzr/Data/skin_xml/cell/data'
    for patch_idx, i in enumerate(os.listdir(os.path.join(path, 'img'))):
        img = cv2.imread(os.path.join(os.path.join(path, 'img'), i))[..., (2, 1, 0)]
        mask = cv2.imread(os.path.join(os.path.join(path, 'label'), i.split('.')[0] + '.png'))[..., (2, 1, 0)]
        yellow = np.logical_and(mask[..., 0], mask[..., 1])
        red = np.logical_xor(mask[..., 0], yellow)
        green = np.logical_xor(mask[..., 1], yellow)
        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].imshow(mask)
        # ax[0, 1].imshow(red)
        # ax[1, 0].imshow(green)
        # ax[1, 1].imshow(yellow)
        # plt.show()
        red_coord = process(red, 0)
        green_coord = process(green, 0)
        coord = [red_coord, green_coord]
        for cell_class, j in enumerate(coord):
            for idx, coordinates in enumerate(j):
                y = int(coordinates[0])
                x = int(coordinates[1])
                ymin = 0 if y - cell_size < 0 else y - cell_size
                ymax = 2047 if y + cell_size > 2047 else y + cell_size
                xmin = 0 if x - cell_size < 0 else x - cell_size
                xmax = 2047 if x + cell_size > 2047 else x + cell_size

                cell = img[ymin:ymax, xmin:xmax, (2, 1, 0)]
                if cell.shape != [cell_size * 2, cell_size * 2]:
                    cell = cv2.resize(cell, (cell_size * 2, cell_size * 2))

                out = os.path.join(outpath, str(cell_class))
                if not os.path.exists(out):
                    os.makedirs(out)
                name = str(patch_idx) + '_' + str(idx) + '.jpg'
                cv2.imwrite(os.path.join(out, name), cell)
                # plt.imshow(cell)
                # plt.show()


if __name__ == '__main__':
    main()