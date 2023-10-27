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

def difference_of_Gaussians(img, k1, s1, k2, s2):
    b1 = cv2.GaussianBlur(img,(k1, k1), s1)
    b2 = cv2.GaussianBlur(img,(k2, k2), s2)
    return b1 - b2

# def make_scan_image(image, width, ksize=(5, 5), min_threshold=75, max_threshold=200):
def make_scan_image(image, width, ksize=(5, 5), min_threshold=180, max_threshold=220):

    image_list_title = []
    image_list = []

    org_image = image.copy()
    image = imutils.resize(image, width=width)
    ratio = org_image.shape[1] / float(image.shape[1])

    # 이미지를 grayscale로 변환하고 blur를 적용
    # 모서리를 찾기위한 이미지 연산
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # for i in range(gray.shape[0]):
    #     for j in range(gray.shape[1]):
    #         if gray[i][j] > 250:
    #             gray[i][j] = 255
    #         elif gray[i][j] <= 250 and gray[i][j] > 245:
    #             gray[i][j] = gray[i][j]*1.02
    #         elif gray[i][j] <= 245 and gray[i][j] > 240:
    #             gray[i][j] = gray[i][j]*1.015
    #         elif gray[i][j] <= 240 and gray[i][j] > 235:
    #             gray[i][j] = gray[i][j]*1.01
    #         elif gray[i][j] <= 235 and gray[i][j] > 230:
    #             gray[i][j] = gray[i][j]*1.005
    #         elif gray[i][j] <= 230 and gray[i][j] > 25:
    #             gray[i][j] = gray[i][j]*1
    #         elif gray[i][j] <= 25 and gray[i][j] > 20:
    #             gray[i][j] = gray[i][j]*1.02
    #         elif gray[i][j] <= 20 and gray[i][j] > 15:
    #             gray[i][j] = gray[i][j]*1.015
    #         elif gray[i][j] <= 15 and gray[i][j] > 10:
    #             gray[i][j] = gray[i][j]*1.01
    #         elif gray[i][j] <= 10 and gray[i][j] > 5:
    #             gray[i][j] = gray[i][j]*1.005
    #         else:
    #             gray[i][j] = 0




    # cv2.imshow('img',gray)
    # cv2.waitKey()

    # Addaptive Gauss Threshholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    # median filter
    median_blur = cv2.medianBlur(thresh, 9)
    # Remove small white point
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(median_blur, None, None, None, 8, cv2.CV_32S)
    # get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:, cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 100:  # keep
            result[labels == i + 1] = 255

    # blurred = cv2.GaussianBlur(result, ksize, 0)
    blurred = difference_of_Gaussians(result, 5,7, 9,9)
    edged = cv2.Canny(blurred, min_threshold, max_threshold)
    # edged = cv2.Canny(gray, min_threshold, max_threshold)
    # cv2.imshow('edged', edged)
    # cv2.waitKey()

    image_list_title = ['gray', 'blurred', 'edged']
    image_list = [gray, blurred, edged]

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

    output = image.copy()
    # print(output.shape)
    cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)
    # print(image.shape[0])
    tmp = np.zeros([image.shape[0],image.shape[1],3],dtype=np.uint8)
    tmp.fill(0)
    cv2.drawContours(tmp, [findCnt], -1, (255, 255, 255), 2)

    cv2.imwrite('black_white/various/true_white/{}'.format(img), tmp)
    # cv2.imwrite('black_white/false/7/{}_black.jpg'.format(img), tmp)
    # cv2.imshow('edged', tmp)
    # cv2.waitKey()



    image_list_title.append("Outline")
    image_list.append(output)

    # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
    transform_image = four_point_transform(org_image, findCnt.reshape(4, 2) * ratio)

    # plt_imshow(image_list_title, image_list)
    # plt_imshow("Transform", transform_image)


    return transform_image


# base_dir = 'C:/Users/USER/PycharmProjects/homography/images/'
base_dir = 'C:/Users/USER/PycharmProjects/homography/black_white/various/true/'
# base_dir = 'C:/Users/USER/PycharmProjects/detect_coordinates_with_yolo/croppedImage/id/'
img_list = os.listdir(base_dir)

for img in img_list:
    org_image = cv2.imread(base_dir+img)

    img_rgb = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
    img_coordinate = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)


    try:
        # receipt_image = make_scan_image(org_image, width=200, ksize=(5, 5), min_threshold=20, max_threshold=100)
        receipt_image = make_scan_image(org_image, width=org_image.shape[1], ksize=(9, 9), min_threshold=20, max_threshold=100)
    except:
        continue

    # cv2.imwrite('cropped/{}_cropped.jpg'.format(img), receipt_image)
    # cv2.imwrite('black_white/true/3/{}.jpg'.format(img), receipt_image)