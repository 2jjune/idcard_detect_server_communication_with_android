import cv2, sys
from matplotlib import pyplot as plt
import numpy as np
import os

base_dir = 'C:/Users/USER/PycharmProjects/homography/data/aaa/'
img_list = os.listdir(base_dir)
def show_img(img):
    cv2.imshow('Corners', img)
    cv2.waitKey()
    # cv2.destroyAllWindows()

def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 120,255, cv2.THRESH_TOZERO)
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    show_img(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    show_img(gray)
    gray = cv2.edgePreservingFilter(gray, flags=1, sigma_s=45, sigma_r=0.2)
    show_img(gray)

    return gray

for img in img_list:
    image = cv2.imread(base_dir+img)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_coordinate = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = edge_detection(image)

    result = cv2.cornerHarris(result, 2, 3, 0.04)
    result = cv2.dilate(result, None, iterations=6)
    img_coordinate[result>0.1*result.max()] = [255,0,0]

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,2,2)
    plt.imshow(img_coordinate)
    plt.xticks([])
    plt.yticks([])

    plt.show()