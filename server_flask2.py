from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import skimage.draw
import imageio
import cv2
import os
import random
from keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Activation, Dropout
from keras.models import Model
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
import matplotlib.pyplot as plt
from io import BytesIO
from urllib import parse
# import cStringIO as StringIO
app = Flask(__name__)


@app.route('/')
def home():
    return '연결 성공!'


# localhost:5000/image
# 갤러리에서 이미지를 보내면 분석을 해서 다시 리턴해주는 부분.
@app.route('/image', methods=['POST', 'GET'])
def image():
    import base64
    global image_list, num

    import json
    # image = request.files['file']
    # imgdata = base64.b64decode(str(image))
    # image_str = request.args.get('image')
    # url_decode = parse.unquote(image_str)
    # image_str = url_decode.replace("data:image/png;base64,", "");
    # image_bytes = base64.b64decode(image_str)
    # 이미지 받아오기-----------------------------------------------------------------
    save_path = './demo_frame/'

    image_str = request.form['image']
    url_decode = parse.unquote(image_str)
    image_str = url_decode.replace("data:image/png;base64,", "");
    image_bytes = base64.b64decode(image_str)

    print(type(image_str))
    print(type(url_decode))
    print(type(image_str))
    print(type(image_bytes))
    print(image_bytes)


    # filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    # with open("imageToSave.jpg", "wb") as fh:
    #     fh.write(base64.decodebytes(imgdata))
    # stream = StringIO.StringIO(image_bytes)
    # img = Image.open(stream)
    # img.show()

    # image_bytes = image_bytes.read()
    im = Image.open(BytesIO(image_bytes))
    # im.show()
    im.save('test.jpg')
    image = cv2.imread('test.jpg')
    cv2.imshow('test',image)
    cv2.waitKey()
    image_list.append(image)
    # image_string = base64.b64encode(image.read())
    # tmp = Image.open(image)
    # tmp.show()
    # tmp.save('test.jpg')
    # image_string 을 분석기에 보내서 분석하는 코드가 필요
    # if image:
    #     filename = secure_filename(image.filename)

    #워핑 모델-------------------------------------------------------------

    # warp_model = build_resnet50_unet((512, 512, 3))
    # warp_model.load_weights('./best_with_RESNET50_512_focalloss_13200_data_shuffle.h5')
    #
    # image = cv2.resize(im, (512, 512))
    # image = np.asarray(image)
    # image = image / 255.
    # image = np.expand_dims(image, axis=0)
    #
    # pred_mask = warp_model.predict(image)
    #
    # pred_mask = np.squeeze(pred_mask, axis=0)
    # pred_mask2 = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 1))
    #
    # for i in range(pred_mask.shape[0]):
    #     for j in range(pred_mask.shape[1]):
    #         if pred_mask[i][j] > 0.5:
    #             pred_mask2[i][j] = 255
    # save_img = tf.keras.utils.array_to_img(pred_mask2)
    #
    # # save_img.save('./segment_result/{0}_{1}.jpg'.format(file[:-4],check_num))
    #
    # save_img = np.array(save_img)
    # save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
    # # cv2.imwrite(f'C:/Users/USER/PycharmProjects/homography/segment_result/webcam/{num}.jpg',save_img)
    # receipt_image = make_scan_image(im, save_img, min_threshold=25, max_threshold=70)
    #
    # if type(receipt_image) == int:
    #     pass
    # else:
    #     # cv2.imwrite(f'C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame/capture_{num}.jpg',receipt_image)
    #     cv2.imwrite(save_path + f'capture_{num}.jpg', receipt_image)
    # del pred_mask, pred_mask2, save_img, receipt_image
    # num += 1
    #-------------------------------------------------------
    #---------------------shift----------------------------
    if len(image_list)==1:
        cvimage1 = image_list[0]
        cv2.imwrite(save_path+f'capture_{num}.jpg',cv2.resize(image_list[num][100:980,295:1695],(512,512)))
        num += 1

    else:
        imgL = image_list[0]
        imgR = image_list[num]

        descriptor = cv2.xfeatures2d.SIFT_create()

        kpsL, featuresL = descriptor.detectAndCompute(imgL, None)
        kpsR, featuresR = descriptor.detectAndCompute(imgR, None)

        matcher = cv2.DescriptorMatcher_create("BruteForce")
        matches = matcher.knnMatch(featuresR, featuresL, 2)

        good_matches = []

        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                good_matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(good_matches)>250:
            ptsL = np.float32([kpsL[i].pt for (i,_) in good_matches])
            ptsR = np.float32([kpsR[i].pt for (_,i) in good_matches])
            matrix, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)

            panorama = cv2.warpPerspective(imgR, matrix, (1980, 1080))
            panorama = panorama[100:980,295:1695]
            panorama= cv2.resize(panorama, (512,512))

            cv2.imwrite(save_path+f'capture_{num}.jpg',panorama)
            num += 1

        else:
            print("warping 실패 다시 찍어주세요.")
            warp = 'no'
            return jsonify(
                warp=warp
            )

    if len(image_list)==5:
    #classification 모델-------------------------------------------------------------
        resnet50 = ResNet50V2(include_top=False, input_shape=(512, 512, 3))

        x = resnet50.output
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(1, activation='sigmoid')(x)

        classification_model = Model(resnet50.input, outputs=x)
        # classification_model.summary()
        classification_model.compile(tf.keras.optimizers.Adagrad(),
                                     loss='binary_crossentropy',
                                     metrics=['accuracy'])
        classification_model.load_weights('./rgb_v4_normalization.h5')

        np_list = make_np_array(save_path)
        img = image_generator_rgb(np_list[0])
        img = np.reshape(img, (1, 512, 512, 3))

        pred = classification_model.predict(img)
        print(pred)


        img = []
        photo = []
        photo_label = []
        second_test_img = []
        # for i in range(0, 5):
        #     img.append(img_list[:, :, int(i + (i * 2)):int(3 + (3 * i))])
        #     img[i] = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
        #     img[i] = cv2.resize(img[i], (256, 256))
        #     img[i] = Image.fromarray(img[i])
        #
        # if pred > 0.5:
        #     fram_pic_dir = './demo_frame_pic/'
        #     if not os.path.exists(fram_pic_dir):
        #         os.makedirs(fram_pic_dir)
        #     delete_files(fram_pic_dir)
        #
        #     for i in range(0, 5):
        #         second_test_img.append(img_list[:, :, int(i + (i * 2)):int(3 + (3 * i))])
        #         #######신분증##########
        #         second_test_img[i][:int(512 * 0.05), :] = 255
        #         second_test_img[i][int(512 * 0.735):, :] = 255
        #         second_test_img[i][:, :int(512 * 0.59)] = 255
        #         second_test_img[i][:, int(512 * 0.985):] = 255
        #
        #         #######면허증##########
        #         # second_test_img[i][:int(512 * 0.236), :] = 255
        #         # second_test_img[i][int(512 * 0.928):, :] = 255
        #         # second_test_img[i][:, :int(512 * 0.018)] = 255
        #         # second_test_img[i][:, int(512 * 0.38):] = 255
        #
        #         cv2.imwrite(fram_pic_dir + f'{i}.jpg', second_test_img[i])
        #     del img
        #     np_list = make_np_array(fram_pic_dir)
        #     img = image_generator_rgb(np_list[0])
        #
        #     img = np.reshape(img, (1, 512, 512, 3))
        #     second_pred = classification_model.predict(img)
        #     print('second pred', second_pred)
        #
        #     if second_pred > 0.4:  # 0.22
        #         ans = '최종 합격'
        #     else:
        #         ans = '사진영역 불합격'
        #
        # else:
        #     ans = '위조 신분증'


        #1. 이미지 저장후 보내기
        # color, shape = server.get_img_inform(model, "C:/Users/user/PycharmProjects/maskrcnn-custom/23.python_server/test.jpg")
        #2. 저장없이 보내기
        # color, shape = server.get_img_inform(model, image)
        ans = '진품'
        return jsonify(
            ans=ans
        )

# localhost:5000/image
# 갤러리에서 이미지를 보내면 분석을 해서 다시 리턴해주는 부분.
# 분석하는 코드는 없음.
@app.route('/image1', methods=['POST', 'GET'])
def image2():
    import base64
    import json
    # image_string 을 분석기에 보내서 분석하는 코드가 필요
    shape = "circle"
    color = "black"
    return jsonify(
        shape=shape,
        color=color
    )

def delete_files(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)

def image_generator_rgb(np_file):
    img = []
    x1 = np.copy(np_file)

    for i in range(5):
        x1[:, :, 3 * (i):3 * (i + 1)] = cv2.cvtColor(x1[:, :, 3 * (i):3 * (i + 1)], cv2.COLOR_BGR2RGB)

    for i in range(x1.shape[2]):
        img.append(np.reshape((x1[:, :, i]), (512, 512, 1)))

    img = np.array(img)
    x1 = np.concatenate((img[0], img[1], img[2], img[3], img[4], img[5], img[6], img[7], img[8], img[9],
                         img[10], img[11], img[12], img[13], img[14]), axis=2)

    x1 = np.float32(x1)

    x_r = x1[:, :, 0::3]
    x_g = x1[:, :, 1::3]
    x_b = x1[:, :, 2::3]

    x_r_mean = np.mean(x_r, axis=2)
    x_g_mean = np.mean(x_g, axis=2)
    x_b_mean = np.mean(x_b, axis=2)

    x_r_std = np.std(x_r, axis=2)
    x_g_std = np.std(x_g, axis=2)
    x_b_std = np.std(x_b, axis=2)

    image_mean = cv2.merge((x_r_mean, x_g_mean, x_b_mean))
    image_std = cv2.merge((x_r_std, x_g_std, x_b_std))

    image = np.concatenate((image_mean, image_std), axis=2)
    image = np.float32(image)

    image[:, :, 0] = image[:, :, 0] / 255.
    image[:, :, 3] = image[:, :, 3] / 255.

    image[:, :, 1] = image[:, :, 1] / 255.
    image[:, :, 4] = image[:, :, 4] / 255.

    image[:, :, 2] = image[:, :, 2] / 255.
    image[:, :, 5] = image[:, :, 5] / 255.

    return image[:, :, 3:6]

def make_np_array(save_path):
    image_name_list = os.listdir(save_path)
    img_num_list = list(range(len(image_name_list)))
    np_list = []
    while (len(img_num_list) >= 5):
        # 이미지 숫자 리스트에서 램덤으로 하나 숫자 꺼내오기
        r1 = img_num_list.pop(random.randrange(0, len(img_num_list)))
        r2 = img_num_list.pop(random.randrange(0, len(img_num_list)))
        r3 = img_num_list.pop(random.randrange(0, len(img_num_list)))
        r4 = img_num_list.pop(random.randrange(0, len(img_num_list)))
        r5 = img_num_list.pop(random.randrange(0, len(img_num_list)))

        # 램덤 이미지 선택
        img = cv2.imread(save_path + image_name_list[r1])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = cv2.imread(save_path + image_name_list[r2])
        # img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(save_path + image_name_list[r3])
        # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img3 = cv2.imread(save_path + image_name_list[r4])
        # img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img4 = cv2.imread(save_path + image_name_list[r5])
        # img4 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = np.concatenate((img, img1, img2, img3, img4), axis=2)
        np_list.append(x)

    return np_list

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
    global img_size
    image = cv2.resize(image, (img_size,img_size))

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
    area = get_area(findCnt.reshape(4,2))
    print(area)

    if area > (image.shape[0]**2)*0.82 or area < (image.shape[0]**2)*0.16 or np.isnan(area) == True:
        return 0
    # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
    findCnt = findCnt.reshape(4,2)
    for i in range(4):
        findCnt[i][0] *= (base_org_image.shape[1]/img_size)
        findCnt[i][1] *= (base_org_image.shape[0]/img_size)
    print(findCnt)
    transform_image = four_point_transform(base_org_image, findCnt)
    transform_image = cv2.resize(transform_image, (img_size, img_size))
    del findCnt,area, approx, peri, cnts, gray, edged

    return transform_image

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_resnet50_unet(input_shape):
    """ Input """
    # inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(input_shape))

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output           ## (512 x 512)
    s2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(resnet50.input, outputs, name="ResNet50_U-Net")
    return model


# 앱에서 사용자가 데이터를 수정해주면 그 정보를 바탕으로 크롤링 하는 부분
# @app.route('/crawl', methods=['POST', 'GET'])
# def crawl():
#     #POST방식
#     # shape = request.form['shape']
#     # color = request.form['color']
#     # text = request.form['text']
#
#     #GET방식
#     shape = request.args.get('shape')
#     color = request.args.get('color')
#     text = request.args.get('text')
#     # image_string 을 분석기에 보내서 분석하는 코드가 필요
#
#     # shape = 'circle'
#     # color = 'white'
#     print(shape, color, text)
#     total = server.crawling_get_link_img_name(shape, color, text)
#
#     # tmp = {'medicine_name' : '인데놀정10mg',
#     #        'medicine_image' : 'https://terms.naver.com/entry.naver?docId=2141123&cid=51000&categoryId=51000',
#     #        'link' : 'https://dbscthumb-phinf.pstatic.net/3323_000_20/20210803233503184_NG6OQL73C.jpg/A11ABBBBB090302.jpg?type=m250&wm=N'}
#     # tmp2 = {'medicine_name' : '소론도정',
#     #         'medicine_image' : 'https://terms.naver.com/entry.naver?docId=2140285&cid=51000&categoryId=51000',
#     #         'link' : 'https://dbscthumb-phinf.pstatic.net/3323_000_9/20171126022112845_RQZOH3G3T.jpg/A11A4290B001503.jpg?type=m250&wm=N'}
#     # total = [tmp,tmp2]
#
#     # 서버에서 string값 두개 리턴하는 코드 여기다가 넣어야함.
#     return jsonify(
#         total = total
#     )


if __name__ == '__main__':
    global image_list, num
    image_list = []
    num=0
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True,host="203.255.93.189",port=5000)
