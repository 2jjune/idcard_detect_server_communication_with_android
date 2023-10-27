import flask
import time
import cv2
import numpy as np
from keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Activation, \
    Dropout
from keras.models import Model
import tensorflow as tf
import os
import random
from PIL import Image
import threading as td

global current_t
global deadline
global threadList
import ctypes

save_path = './demo_frame/'
photo_area_pic_dir = './photo_area_pic/'
app = flask.Flask(__name__)
image_list = []
threadList = []

deadline = 30

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
    print("Server is connect")
    deleteAllFile([save_path, photo_area_pic_dir])
    return "Server is connect"


@app.route('/ok', methods=['GET', 'POST'])
def OK():
    print('thread id:', td.get_ident())
    threadList.append(td.get_ident())

    for i in range(deadline):
        print('times go.. ' + str(td.get_ident()) + ': ' + str(i))
        time.sleep(1)
        if i == (deadline - 1):
            print("the Server is restart")
            deleteAllFile([save_path, photo_area_pic_dir])
            return "200"


def raise_exception(id):
    print("the thread id killed is : ", id)
    thread_id = id
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
                                                     ctypes.py_object(SystemExit))
    global threadList
    threadList.pop(0)

    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        print('Exception raise failure')


# the post method. when we call this with a string containing a name, it will return the name with the text "I got your name"
@app.route('/getname/<name>', methods=['POST'])
def extract_name(name):
    return "I got your name " + name;


@app.route('/getfact', methods=['GET'])
def getfact(msg):
    return msg


def warping_classification(img, imagefile):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(photo_area_pic_dir):
        os.makedirs(photo_area_pic_dir)

    imageName = str(imagefile).split(" ")[1].replace("'", "")
    index = imageName.split("_")[2]

    # imagePath ='C:/Users/USER/PycharmProjects/homography/'+imageName
    # image = cv2.imread(imagePath)
    image_list.append(img)
    num = len(image_list)
    w = img.shape[1]
    h = img.shape[0]

    print("num --->", num)

    if (num - 1) == 4:
        imgL = image_list[0]
        imgR = image_list[num - 1]

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

            panorama = cv2.warpPerspective(imgR, matrix, (1920, 1080))
            panorama = panorama[int(h * 0.2):int(h * 0.8), int(w * 0.15):int(w * 0.835)]
            # panorama = panorama[int(h*0.154):int(h*0.883),int(w*0.093):int(w*0.91)]
            panorama = cv2.resize(panorama, (512, 512))

            cv2.imwrite(save_path + f'capture_{index}', panorama)


        else:
            print("warping failed")
            warp = 'Try Again_' + str(num - 1)
            image_list.pop()

            return warp

        print("imageList==5")

        np_list = make_np_array(save_path)
        img = image_generator_rgb(np_list[0])
        img = np.reshape(img, (1, 512, 512, 3))

        pred = classification_model.predict(img)
        print(pred)
        if pred > 0.5:
            # /////////////////////for second prediction//////////////////////////
            second_test_img = []

            for i in range(0, 5):
                second_test_img.append(np_list[0][:, :, int(i + (i * 2)):int(3 + (3 * i))])
                #######면허증##########
                second_test_img[i][:int(512 * 0.236), :] = 255
                second_test_img[i][int(512 * 0.928):, :] = 255
                second_test_img[i][:, :int(512 * 0.018)] = 255
                second_test_img[i][:, int(512 * 0.38):] = 255

                cv2.imwrite(photo_area_pic_dir + f'{i}.jpg', second_test_img[i])

            np_list = make_np_array(photo_area_pic_dir)
            img = image_generator_rgb(np_list[0])
            img = np.reshape(img, (1, 512, 512, 3))
            second_pred = classification_model.predict(img)
            print('second pred : ', second_pred)
            if second_pred > 0.4:
                ans = 'This id card is real_0'
            else:
                ans = 'The photo area is Fake_0'

            # ////////////////////////////////////////////////////////////////

            # ans = 'This id card is real_0'
        else:
            ans = 'This iD card is Fake_0'

        deleteAllFile([save_path, photo_area_pic_dir])

        return ans

    if (num - 1) == 0:
        cv2.imwrite(save_path + f'capture_{index}',
                    cv2.resize(img[int(h * 0.2):int(h * 0.8), int(w * 0.15):int(w * 0.835)], (512, 512)))
        # cv2.imwrite(save_path+f'capture_{index}.jpg',cv2.resize(img[int(h*0.154):int(h*0.883),int(w*0.093):int(w*0.91)],(512,512)))

        return "the first image is taken_1"

    elif (num - 1) > 0:
        imgL = image_list[0]
        imgR = image_list[num - 1]

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

            panorama = cv2.warpPerspective(imgR, matrix, (1920, 1080))
            panorama = panorama[int(h * 0.2):int(h * 0.8), int(w * 0.15):int(w * 0.835)]
            # panorama = panorama[int(h*0.154):int(h*0.883),int(w*0.093):int(w*0.91)]
            panorama = cv2.resize(panorama, (512, 512))

            cv2.imwrite(save_path + f'capture_{index}', panorama)
            warp = 'yes_' + str(num)
        else:
            print("warping failed")
            image_list.pop()
            warp = 'Try Again_' + str(num - 1)

        return warp


@app.route('/', methods=['GET', 'POST'])
def handle_request():
    print("im in handle_request")
    if len(threadList) > 0:
        print("thread is is : ", threadList[0])
        raise_exception(threadList[0])
    imagefile = flask.request.files['image']
    # filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    img = Image.open(imagefile)
    print(img.size)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    Result = warping_classification(img, imagefile)

    return Result


def deleteAllFile(path_list):
    global image_list
    for path in path_list:
        if os.path.exists(path):
            for file in os.scandir(path):
                print(file.path)
                image_list.clear()
                os.remove(file.path)
            print("All file is Deleted Your Server is Ready Now!!")


# this commands the script to run in the given port
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)  # use_reloader:모듈변경시재실행,


