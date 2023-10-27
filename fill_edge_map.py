import cv2
import numpy as np
from pprint import pprint
import os

# image = 'C:/Users/USER/PycharmProjects/img_dataset_for_edge_detection/edge_maps/train/false_cc4_jpg.rf.d91bd20a4dba6f86ea05676a2e12e4a7.jpg'

base_dir = 'C:/Users/USER/PycharmProjects/homography/segment_result/vidoes_frame_4_train_co/'
img_list = os.listdir(base_dir)

for image in img_list:
    img = cv2.imread(base_dir + image)
    th, im_th = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
    img_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    print(h, w)
    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(img_floodfill, mask, (0,0), 0)

    cv2.imwrite('C:/Users/USER/PycharmProjects/homography/segment_result/vidoes_frame_4_train_co_filled/{}'.format(image),img_floodfill)



# cv2.imshow('threshold', im_th)
# cv2.imshow('floodfilled', img_floodfill)
# cv2.imshow('inverted floodfilled', img_floodfill_inv)
# cv2.imshow('foreground', im_out)
# cv2.waitKey()

# img_floodfill = cv2.resize(img_floodfill,(10,10))
# pprint(img_floodfill)
#
# cv2.imshow('floodfilled', img_floodfill)
# cv2.waitKey()
