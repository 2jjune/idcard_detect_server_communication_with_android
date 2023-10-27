from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
import pytesseract
import imutils
import cv2
import re
import requests
import numpy as np
import os


# base_dir = 'C:/Users/USER/PycharmProjects/homography/black_white/true/'
base_dir = 'C:/Users/USER/PycharmProjects/homography/segment_result/videos_frame_4_train_seperate/0/'
compare_dir = 'C:/Users/USER/PycharmProjects/homography/segment_result/vidoes_frame_4_train_co_filled/val/'

img_list = os.listdir(base_dir)
compare_list = os.listdir(compare_dir)

for img in img_list:
    # cv2.imwrite('C:/Users/USER/PycharmProjects/DexiNed/id_data/edges/edge_maps/train/rgbr/real/true_{}.jpg'.format(img[:-4]), org_image)

    for compare in compare_list:
        # print(img)
        # print(compare)
        if img == compare:
            org_image = cv2.imread(base_dir + img)
            # print(img)
            # print('true')
            cv2.imwrite('C:/Users/USER/PycharmProjects/homography/segment_result/videos_frame_4_train_seperate/val/{}.jpg'.format(img[:-4]), org_image)
            break

        # img_rgb = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

        # print(img[:-4])
        # cv2.imwrite('C:/Users/USER/PycharmProjects/DexiNed/id_data/edges/edge_maps/train/rgbr/real/true_{}.jpg'.format(img[:-4]), org_image)
        # cv2.waitKey()