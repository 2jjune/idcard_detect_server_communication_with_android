import cv2, sys
from matplotlib import pyplot as plt
import numpy as np
import os

base_dir = 'C:/Users/USER/PycharmProjects/homography/data/aaa/'
img_list = os.listdir(base_dir)

for img in img_list:
    image = cv2.imread(base_dir+img)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_coordinate = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # blur = cv2.GaussianBlur(img_gray, ksize=(7, 7), sigmaX=0)
    # ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    # edged = cv2.Canny(blur, 10, 250)
    #
    # image_gray = np.float32(edged)
    # result = cv2.cornerHarris(img_gray, 2, 3, 0.04)
    # result = cv2.dilate(result, None, iterations=6)
    # img_coordinate[result>0.1*result.max()] = [255,0,0]
    #
    # plt.subplot(1,2,1)
    # plt.imshow(img_rgb)
    # plt.xticks([])
    # plt.yticks([])
    #
    # plt.subplot(1,2,2)
    # plt.imshow(img_coordinate)
    # plt.xticks([])
    # plt.yticks([])
    #
    # plt.show()

    corners = cv2.goodFeaturesToTrack(img_gray, 80, 0.01, 10)
    # 실수 좌표를 정수 좌표로 변환
    corners = np.int32(corners)

    # 좌표에 동그라미 표시
    for corner in corners:
        x, y = corner[0]
        cv2.circle(image, (x, y), 5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('Corners', image)
    cv2.waitKey()
    cv2.destroyAllWindows()