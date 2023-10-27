import cv2, sys
from matplotlib import pyplot as plt
import numpy as np
import os
import math
import os
import random
from keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Activation, Dropout
from keras.models import Model
import cv2
import numpy as np
import tensorflow as tf
from imutils.perspective import four_point_transform
import imutils
import math
import tkinter as tk
from tkinter import Tk, Label, PhotoImage, Button
import tkinter.font as font
from PIL import Image, ImageTk
from tensorflow.keras.layers import Conv2DTranspose, Concatenate
from datetime import datetime

# base_dir = 'C:/Users/USER/PycharmProjects/homography/data/aaa/'
# img_list = os.listdir(base_dir)
# # print(img_list)
#
# for img in img_list:
#     image = cv2.imread(base_dir+img)
#     image_gray = cv2.imread(base_dir+img, cv2.IMREAD_GRAYSCALE)
#
#     blur = cv2.GaussianBlur(image_gray, ksize=(7,7), sigmaX=0)
#     ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
#     edged = cv2.Canny(blur, 10, 250)
#     # cv2.imshow('Edged', edged)
#     # cv2.waitKey(0)
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
#     closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
#     # cv2.imshow('closed', closed)
#     # cv2.waitKey(0)
#
#     contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     total = 0
#     contours_image = cv2.drawContours(image, contours, -1, (0,255,0), 3)
#     # cv2.imshow('contours_image', contours_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     contours_xy = np.array(contours)
#     print(contours_xy.shape)
#     print(len(contours)) #외곽선 개수
#     print(contours[0])
#     cv2.imshow('contours_image', contours_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # x_min, x_max = 0, 0
#     # value = list()
#     # for i in range(len(contours_xy)):
#     #     for j in range(len(contours_xy[i])):
#     #         value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
#     #         x_min = min(value)
#     #         x_max = max(value)
#     # print(x_min)
#     # print(x_max)
#     #
#     # # y의 min과 max 찾기
#     # y_min, y_max = 0, 0
#     # value = list()
#     # for i in range(len(contours_xy)):
#     #     for j in range(len(contours_xy[i])):
#     #         value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
#     #         y_min = min(value)
#     #         y_max = max(value)
#     # print(y_min)
#     # print(y_max)
#     # x = x_min
#     # y = y_min
#     # w = x_max-x_min
#     # h = y_max-y_min
#     #
#     # img_trim = image[y:y+h, x:x+w]
#     # cv2.imwrite('cropped/{}_cropped.jpg'.format(img), img_trim)
#     # org_image = cv2.imread('org_trim.jpg')
#     # cv2.imshow('org_image', org_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#
# a1 = (1,1)
# a2 = (2,4)
# a3 = (3,1)
# a4 = (5,3)
#
# o1 = math.atan((a2[1]-a1[1])/(a2[0]-a1[0]))
# o2 = math.atan((a3[1]-a1[1])/(a3[0]-a1[0]))
# angle1 = abs((o1-o2)*180/math.pi)
#
# o3 = math.atan((a2[1]-a4[1])/(a2[0]-a4[0]))
# o4 = math.atan((a3[1]-a4[1])/(a3[0]-a4[0]))
# angle2 = abs((o3-o4)*180/math.pi)
#
# area = (0.5 * math.sqrt((a2[0]-a1[0])**2+(a2[1]-a1[1])**2) * math.sqrt((a3[0]-a1[0])**2+(a3[1]-a1[1])**2) * math.sin(math.radians(angle1)))/
#        +(0.5 * math.sqrt((a2[0]-a4[0])**2+(a2[1]-a4[1])**2) * math.sqrt((a3[0]-a4[0])**2+(a3[1]-a4[1])**2) * math.sin(math.radians(angle2)))
# print(angle1, angle2)
# # print(0.5 * math.sqrt((a2[0]-a1[0])**2+(a2[1]-a1[1])**2) * math.sqrt((a3[0]-a1[0])**2+(a3[1]-a1[1])**2))
# # print(0.5 * math.sqrt((a2[0]-a4[0])**2+(a2[1]-a4[1])**2) * math.sqrt((a3[0]-a4[0])**2+(a3[1]-a4[1])**2))
# # print(math.sin(math.radians(angle1)))
# # print(math.sin(math.radians(angle2)))
# print(area)
#
#

# file_path = 'C:/Users/USER/PycharmProjects/img_dataset_for_edge_detection/moremore/19/'
# file_names = os.listdir(file_path)
#
# for name in file_names:
#     src = os.path.join(file_path, name)
#     dst = 'more_' + name
#     dst = os.path.join(file_path, dst)
#     os.rename(src, dst)



# students = [('강타우','김현민'),('박준원','이진택'),('조유진','문은식'),('박선호','구창규'),('배인기','김영권'),('이현태','양승민'),
#             ('문수인','최혜안'),('차재혁','김성혁'),('전혜연','이화진'),('김단희','심은아'),('최수빈','조하영'),('김승민','하준서'),
#             ('이승준','오준석'),('박재민','이현영'),('조정은','최재훈')]
#
# day_1_9 = []
# day_1_10 = []
# day_1_11 = []
# day_2_9 = []
# day_2_10 = []
# day_2_11 = []
#
# day_1_9.append(students.pop(random.randrange(0, 2)))
# day_1_9.append(students.pop(random.randrange(0, 2)))
# day_1_10.append(students.pop(random.randrange(0, 2)))
# day_1_10.append(students.pop(random.randrange(0, 2)))
# day_1_11.append(students.pop(random.randrange(0, 2)))
# day_1_11.append(students.pop(random.randrange(0, 2)))
# day_2_9.append(students.pop(random.randrange(0, 2)))
# day_2_9.append(students.pop(random.randrange(0, 2)))
# day_2_10.append(students.pop(random.randrange(0, 2)))
# day_2_10.append(students.pop(random.randrange(0, 2)))
# day_2_11.append(students.pop(random.randrange(0, 2)))
# day_2_11.append(students.pop(random.randrange(0, 2)))
#
# day_1_910 = students.pop(random.randrange(0, 1))
# day_1_11_day2_9 = students.pop(random.randrange(0, 1))
# day_2_1011 = students.pop(random.randrange(0, 1))
#
#
# print('첫째 9시반: ',day_1_9)
# print('첫째 10시반: ',day_1_10)
# print('첫째 11시반: ',day_1_11)
# print('둘째 9시반: ',day_2_9)
# print('둘째 10시반: ',day_2_10)
# print('둘째 11시반: ',day_2_11)
# print('첫째 9시반-10시반: ',day_1_910)
# print('첫째 11시반-둘째9시반: ',day_1_11_day2_9)
# print('둘째 10시반-11시반: ',day_2_1011)
#
#
def solution(dots):
    for i in range(4):
        for j in range(1,4):
            print(i, j)
            dot = dots.copy()
            tmp = abs((dot[j][1]-dot[i][1])/(dot[j][0]-dot[i][0]))
            dot.pop(i)
            dot.pop(j-1)
            print(tmp)
            if tmp == abs((dot[1][1]-dot[0][1])/(dot[1][0]-dot[0][0])):
                return 1
    return 0

solution([[3, 5], [4, 1], [2, 4], [5, 10]])