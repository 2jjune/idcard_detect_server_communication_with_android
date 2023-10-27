import flask
import requests

import werkzeug
import time
import cv2
import numpy as np
from keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Activation, Dropout
from keras.models import Model
import tensorflow as tf
import os
import random

app = flask.Flask(__name__)
image_list = []


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


# the get method. when we call this, it just return the text "Hey!! I'm the fact you got!!!"
@app.route('/getfact', methods=['GET'])
def get_fact():
    return "Hey!! I'm the fact you got!!!"


# the post method. when we call this with a string containing a name, it will return the name with the text "I got your name"
@app.route('/getname/<name>', methods=['POST'])
def extract_name(name):
    return "I got your name " + name;


@app.route('/', methods=['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    # ///////////////////////////////////////////
    save_path = './demo_frame/'
    global image_list

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imageName=str(imagefile).split(" ")[1].replace("'","")
    imagePath ='D:/androCard/'+imageName
    index=imageName.split("_")[2]

    image = cv2.imread(imagePath)
    image_list.append(image)
    w=image.shape[1]
    h=image.shape[0]

    num = len(image_list)
    if len(os.listdir(save_path))==4:
        imgL = image_list[0]
        imgR = image_list[num-1]

        descriptor = cv2.xfeatures2d.SIFT_create()

        kpsL, featuresL = descriptor.detectAndCompute(imgL, None)
        kpsR, featuresR = descriptor.detectAndCompute(imgR, None)

        matcher = cv2.DescriptorMatcher_create("BruteForce")
        matches = matcher.knnMatch(featuresR, featuresL, 2)

        good_matches = []

        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                good_matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(good_matches) > 250:
            ptsL = np.float32([kpsL[i].pt for (i, _) in good_matches])
            ptsR = np.float32([kpsR[i].pt for (_, i) in good_matches])
            matrix, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)

            panorama = cv2.warpPerspective(imgR, matrix, (1980, 1080))
            panorama = panorama[100:w-1,295:h-1]
            panorama = cv2.resize(panorama, (512, 512))

            cv2.imwrite(save_path+f'capture_{index}.jpg',panorama)
        else:
            print("warping failed")
            warp = 'no'
            return warp
            
        print("imageList==5")
        
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
        classification_model.summary()
        classification_model.compile(tf.keras.optimizers.Adagrad(),
                                     loss='binary_crossentropy',
                                     metrics=['accuracy'])
        classification_model.load_weights('./rgb_v4_normalization.h5')

        np_list = make_np_array(save_path) 
        img = image_generator_rgb(np_list[0])
        img = np.reshape(img, (1, 512, 512, 3))

        pred = classification_model.predict(img)
        print(pred)
        if pred>0.5:
            ans = 'true'
        else:
            ans = 'false'

        if os.path.exists(save_path):
            for file in os.scandir(save_path):
                os.remove(file.path)
        image_list.clear()

        return ans
        
    if len(os.listdir(save_path)) == 0:
        #cv2.imwrite(save_path+f'capture_{num}.jpg',cv2.resize(imagename[100:767,295:1023],(512,512)))
        cv2.imwrite(save_path+f'capture_{index}.jpg',cv2.resize(image[100:w-1,295:h-1],(512,512)))
        return "Go"
        
    elif len(os.listdir(save_path))>0:
        print("imageList>1")
        imgL = image_list[0]
        imgR = image_list[num-1]

        descriptor = cv2.xfeatures2d.SIFT_create()

        kpsL, featuresL = descriptor.detectAndCompute(imgL, None)
        kpsR, featuresR = descriptor.detectAndCompute(imgR, None)

        matcher = cv2.DescriptorMatcher_create("BruteForce")
        matches = matcher.knnMatch(featuresR, featuresL, 2)

        good_matches = []

        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                good_matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(good_matches) > 250:
            ptsL = np.float32([kpsL[i].pt for (i, _) in good_matches])
            ptsR = np.float32([kpsR[i].pt for (_, i) in good_matches])
            matrix, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 4.0)

            panorama = cv2.warpPerspective(imgR, matrix, (1980, 1080))
            panorama = panorama[100:w-1,295:h-1]
            panorama = cv2.resize(panorama, (512, 512))

            cv2.imwrite(save_path+f'capture_{index}.jpg',panorama)
            warp = 'yes'
        else:
            print("warping failed")
            warp = 'no'
        return warp

    

    #return "Go"
    # /////////////////////////////////////////////////////

    # return "Image(s) Uploaded Successfully. Come Back Soon."


# this commands the script to run in the given port
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)