import matplotlib.pyplot as plt
import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
# from tensorflow.keras.prerprocessing.image import ImageDataGenerator
# from tensorflow import keras
import edge_detection_model
from PIL import Image
from imutils.perspective import four_point_transform
import imutils
import math

img_size = 512
model = edge_detection_model.build_resnet50_unet((img_size, img_size, 3))
model.load_weights('./checkpoint/best_with_RESNET50_512_focalloss_8900_data_shuffle.h5')

# for aaaa in range(2,21):
base_dir = 'C:/Users/USER/PycharmProjects/homography/segment_result/bin4_inchang/11/'
dataset = os.listdir(base_dir)
# print(dataset[0])
# org_image = 'test_img13.jpg'

videoPath = 'C:/Users/USER/PycharmProjects/homography/segment_result/videos/'
# imagePath = 'C:/Users/user/Desktop/frame/'
video_list = os.listdir(videoPath)
frames = []

def get_area(findCnt):
    left_up = findCnt[0]
    left_down = findCnt[1]
    right_up = findCnt[2]
    right_down = findCnt[3]

    o1 = math.atan((left_up[1] - left_down[1]) / (left_up[0] - left_down[0]))
    o2 = math.atan((right_down[1] - left_down[1]) / (right_down[0] - left_down[0]))
    angle1 = abs((o1 - o2) * 180 / math.pi)

    o3 = math.atan((left_up[1] - right_up[1]) / (left_up[0] - right_up[0]))
    o4 = math.atan((right_down[1] - right_up[1]) / (right_down[0] - right_up[0]))
    angle2 = abs((o3 - o4) * 180 / math.pi)

    area = (0.5 * math.sqrt((left_up[0] - left_down[0]) ** 2 + (left_up[1] - left_down[1]) ** 2) * math.sqrt(
        (right_down[0] - left_down[0]) ** 2 + (right_down[1] - left_down[1]) ** 2) * math.sin(math.radians(angle1))) \
           + (0.5 * math.sqrt((left_up[0] - right_up[0]) ** 2 + (left_up[1] - right_up[1]) ** 2) * math.sqrt(
        (right_down[0] - right_up[0]) ** 2 + (right_down[1] - right_up[1]) ** 2) * math.sin(math.radians(angle2)))

    return area


def make_scan_image(base_org_image, image, min_threshold=180, max_threshold=220):
    org_image = image.copy()
    # image = imutils.resize(image, height=width, width=width)
    image = cv2.resize(image, (img_size,img_size))
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
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
        if len(approx) == 4:
            findCnt = approx
            break
    # 만약 추출한 윤곽이 없을 경우 오류
    if findCnt is None:
        return 0
        # raise Exception(("Could not find outline."))
    # print(findCnt.reshape(4,2)[0][0])
    area = get_area(findCnt.reshape(4,2))
    # print(findCnt.reshape(4,2))
    print(area)
    # print(np.isnan(area))
    # output = image.copy()
    # cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)
    # print(image.shape[0])

    # print(findCnt)
    # cv2.imwrite('C:/Users/USER/PycharmProjects/homography/segment_result/warped/test.jpg', tmp)
    # cv2.imshow('org', image)
    # cv2.imshow('edged', output)
    # cv2.waitKey()

    if area > (image.shape[0]**2)*0.8 or area < (image.shape[0]**2)*0.2 or np.isnan(area) == True:
        return 0
    # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
    # print(base_org_image.shape)
    findCnt = findCnt.reshape(4,2)
    for i in range(4):
        findCnt[i][0] *= (base_org_image.shape[1]/img_size)
        findCnt[i][1] *= (base_org_image.shape[0]/img_size)
    print(findCnt)
    transform_image = four_point_transform(base_org_image, findCnt)
    transform_image = cv2.resize(transform_image, (img_size, img_size))

    return transform_image


for file in video_list:
    print(file)
    if not (os.path.isdir(videoPath + file)):
        # os.makedirs(os.path.join(imagePath + file))
        cap = cv2.VideoCapture(videoPath + file)
        count = 0
        while True:
            ret, image = cap.read()
            if not ret:
                break
            if count % 2 == 0:
                # cv2.imwrite(imagePath + file + "/frame%d.jpg" % count, image)
                frames.append(image)
            # print('%d.jpg done' % count)
            count += 1
        print(len(frames))
        cap.release()

    check_num = 0
    for org_image in frames:
        # print(img)
        # org_image = cv2.imread(base_dir + img)
        # org_image = cv2.resize(org_image, (img_size, img_size))
        # cv2.imwrite('./segment_result/org/{}'.format(img), image)
        image = cv2.resize(org_image,(img_size, img_size))
        # print(org_image.shape)
        # print(image.shape)
        image = np.asarray(image)
        image = image/255.
        image = np.expand_dims(image, axis=0)

        pred_mask = model.predict(image)

        # image = np.squeeze(image, axis=0)
        pred_mask = np.squeeze(pred_mask, axis=0)
        # print(pred_mask.shape)
        pred_mask2 = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 1))
        # print(pred_mask2.shape)

        for i in range(pred_mask.shape[0]):
            for j in range(pred_mask.shape[1]):
                # print(pred_mask[i][j])
                if pred_mask[i][j]>0.5:
                    pred_mask2[i][j]=255
        # print(pred_mask[150][150])
        save_img = tf.keras.utils.array_to_img(pred_mask2)
        save_img.save('./segment_result/{0}_{1}.jpg'.format(file[:-4],check_num))

        save_img = np.array(save_img)
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)

        receipt_image = make_scan_image(org_image, save_img, min_threshold=25, max_threshold=70)
        # print(type(receipt_image))
        if type(receipt_image) == int:
            continue
        else:
            cv2.imwrite('C:/Users/USER/PycharmProjects/homography/segment_result/final_result_inchang/{0}_{1}.jpg'.format(file[:-4],check_num), receipt_image)
            check_num += 1

