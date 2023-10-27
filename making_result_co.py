import matplotlib.pyplot as plt
import cv2
import requests
import numpy as np
import os

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
# from tensorflow.keras.prerprocessing.image import ImageDataGenerator
# from tensorflow import keras
import edge_detection_model
from PIL import Image

# base_dir = 'C:/Users/USER/PycharmProjects/homography/segment_result/segmentated/'
# dataset = os.listdir(base_dir)
#
# src = cv2.imread(base_dir + dataset[0])
# # src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# # ret,thresh_img = cv2.threshold(imgray, 100, 255, cv2.THRESH_BINARY)
# #
# # contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # # print(len(contours[0]))
# # # cnt = contours[0][4]
# # # cv2.drawContours(image, [cnt], 0, (0,255,0), 3)
# # cv2.drawContours(image, contours, -1, (0,255,0), 3)
# # cv2.imshow('aa',image)
# # cv2.waitKey()
#
# # corners = cv2.goodFeaturesToTrack(src, 400, 0.01, 10)
# #
# # dst1 = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
# #
# # if corners is not None:
# #     for i in range(corners.shape[0]):  # 코너 갯수만큼 반복문
# #         pt = (int(corners[i, 0, 0]), int(corners[i, 0, 1]))  # x, y 좌표 받아오기
# #         cv2.circle(dst1, pt, 5, (0, 0, 255), 2)  # 받아온 위치에 원
# #
# # # Fast 코너 검출
# # fast = cv2.FastFeatureDetector_create(1000)  # 임계값 60 지정
# # keypoints = fast.detect(src)  # Keypoint 객체를 리스트로 받음
# #
# # dst2 = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
# #
# # for kp in keypoints:
# #     pt = (int(kp.pt[0]), int(kp.pt[1]))  # kp안에 pt좌표가 있음
# #     cv2.circle(dst2, pt, 5, (0, 0, 255), 2)
# #
# # cv2.imshow('src', src)
# # cv2.imshow('dst1', dst1)
# # cv2.imshow('dst2', dst2)
# # cv2.waitKey()
# img2 = src.copy()
#
# src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# src = np.float32(src)
# dst = cv2.cornerHarris(src, 2, 3, 0.06)
# # dst = cv2.dilate (dst, None)
# img2[dst>0.01*dst.max()] = [0,0,255]
#
# cv2.imshow('dst2', dst)
# cv2.waitKey()
# print(dst.shape)

from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
import imutils
import cv2
import re
import requests
import numpy as np
import os


def plt_imshow(title='image', img=None, figsize=(8, 5)):
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()


# def make_scan_image(image, width, ksize=(5, 5), min_threshold=75, max_threshold=200):
def make_scan_image(base_org_image, image, width, ksize=(5, 5), min_threshold=180, max_threshold=220):

    image_list_title = []
    image_list = []

    org_image = image.copy()
    image = imutils.resize(image, height=width, width=width)
    ratio = org_image.shape[1] / float(image.shape[1])

    # 이미지를 grayscale로 변환하고 blur를 적용
    # 모서리를 찾기위한 이미지 연산
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, min_threshold, max_threshold)

    # contours를 찾아 크기순으로 정렬
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    findCnt = None
    # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
        if len(approx) == 4:
            findCnt = approx
            break
    # 만약 추출한 윤곽이 없을 경우 오류
    if findCnt is None:
        raise Exception(("Could not find outline."))

    # output = image.copy()
    # cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)
    # print(image.shape[0])

    # print(findCnt)
    # cv2.imwrite('C:/Users/USER/PycharmProjects/homography/segment_result/warped/test.jpg', tmp)
    # cv2.imshow('org', image)
    # cv2.imshow('edged', output)
    # cv2.waitKey()

    # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
    transform_image = four_point_transform(base_org_image, findCnt.reshape(4, 2) * ratio)
    transform_image = cv2.resize(transform_image, (512, 512))

    return transform_image

def main():
    base_dir = 'C:/Users/USER/PycharmProjects/homography/segment_result/segmentated/'
    org_dir = 'C:/Users/USER/PycharmProjects/homography/segment_result/org/'
    img_list = os.listdir(base_dir)

    for img in img_list:
        org_image = cv2.imread(org_dir+img)
        segment_image = cv2.imread(base_dir+img)
        # img_rgb = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
        # img_gray = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
        # img_coordinate = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

        try:
            # receipt_image = make_scan_image(org_image, width=200, ksize=(5, 5), min_threshold=20, max_threshold=100)
            receipt_image = make_scan_image(org_image, segment_image, width=512, ksize=(9, 9), min_threshold=25, max_threshold=70)

        except:
            continue
        cv2.imwrite('C:/Users/USER/PycharmProjects/homography/segment_result/coordinates/{}'.format(img), receipt_image)
main()