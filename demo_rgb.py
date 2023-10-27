import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from PIL import Image, ImageTk
from keras.applications.resnet_v2 import ResNet50V2
from keras import layers
from keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Activation, Dropout
from tensorflow.keras.regularizers import l2
from keras.models import Model
import demo
import demo_webcam
import edge_detection_model
# from tkinter import *
from tkinter import Tk, Label, PhotoImage, Button
import tkinter as tk
import tkinter.font as font
import PIL
import demo_webcam_v2


def delete_files(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)

def make_np_array(save_path):
    # 이미지 이름 리스트
    image_name_list = []
    # 이미지 숫자 리스트
    img_num_list = []
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

    # print(x_v)
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

# def encoding_theshold(pred):
#     for i in range(len(pred)):
#         if pred[i] >= threshold:
#             pred[i] = 1
#         elif pred[i] < threshold:
#             pred[i] = 0
#
#     return pred

def close_app():
   global win, window
   win.destroy()
   # window.destroy()
   # tmp = App().check_continue_no()

class check_finish:
    tmp = 0
    def __init__(self):
        global win, window
        # win = Tk()
        # win.title("Confirm")
        label=tk.Label(win,text='계속 진행하시겠습니까?', font=35)
        # label.place(x=0,y=0)
        # label.pack(side='bottom')
        label.place(x=595, y=430)


        btn = Button(win, text="예", command=self.check_continue_yes, width=5)
        btn2 = Button(win, text="아니오", command=close_app, width=5)
        # btn.pack(padx=12)
        # btn2.pack(padx=10)
        btn.place(x=600, y=450)
        btn2.place(x=700, y=450)
        # btn.pack(side='left')
        # btn2.pack(side='left')

        # width = win.winfo_screenwidth()
        # height = win.winfo_screenheight()
        # print(width, height)

        win.mainloop()

    def __repr__(self):
        return "% s" % (self.tmp)
        # elif "no":
        #     return "% s" % (self.tmp)

    # def __str__(self):
    #     return "a:% s b:% s" % (self.yes, self.no)

    def check_continue_yes(self):
        self.tmp = 1
        win.destroy()
        window.destroy()
        return self.tmp

def print_result(img_list, pred):
    global win, window
    # win.update()
    # window = Tk()
    # window.update()
    win = Tk()
    win.title("결과")
    win.geometry("1350x500")

    img = []
    photo = []
    photo_label = []
    for i in range(0,5):
        img.append(img_list[:,:,int(i+(i*2)):int(3+(3*i))])
        img[i] = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
        img[i] = cv2.resize(img[i], (256, 256))
        img[i] = Image.fromarray(img[i])
        photo.append(ImageTk.PhotoImage(image=img[i]))
        photo_label.append(Label(win, image=photo[i]))
        photo_label[i].pack(side='left')
        photo_label[i].pack(padx=5)

    if pred>0.5:
        label = tk.Label(win, text='진품입니다', font=font.Font(size=20, weight='bold', underline=1), background='white')
    else:
        label = tk.Label(win, text='위조 신분증입니다', font=font.Font(size=20, weight='bold', underline=1), background='white')
    # label.pack(side='top')
    label.place(x=600, y=390)

    ans = check_finish()
    # btn = Button(win, text="확인", command=close_app, width=10, height=3)
    # btn.pack(side='bottom', )

    win.mainloop()
    return ans


def main():
    save_path = 'C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame/'

    threshold = 0.5
    img_height, img_width, input_channel = 512, 512, 3

    webcam_model = edge_detection_model.build_resnet50_unet((512, 512, 3))
    webcam_model.load_weights('./checkpoint/best_with_RESNET50_512_focalloss_8900_data_shuffle.h5')

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

    model = Model(resnet50.input, outputs=x)
    model.summary()
    model.compile(tf.keras.optimizers.Adagrad(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.load_weights('./rgb.h5')

    while(1):
        delete_files('C:/Users/USER/PycharmProjects/homography/segment_result/webcam/')
        delete_files('C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame/')
        # demo.main(webcam_model)

        # q = demo_webcam.main(webcam_model)
        q = demo_webcam_v2.main(webcam_model)
        if q=='q':
            break


        np_list = make_np_array(save_path)
        # print(len(np_list))

        # for i in range(len(np_list)):
        img = image_generator_rgb(np_list[0])
        # print(np_list[i].shape)
        # cv2.imshow('img', np_list[i][:,:,:3])
        # cv2.imshow('aaa',img)
        # cv2.imshow('aaa2',np_list[i])
        # cv2.waitKey()
        img = np.reshape(img,(1,512,512,3))
        pred = model.predict(img)
        print(pred)
        ans = print_result(np_list[0], pred)
        # if pred >= 0.5:
        #     print('진짜 신분증')
        # else:
        #     print("가짜 신분증")



        # ans = check_finish()

        # print('\n')
        # print(ans)
        # print(dir(ans))
        # print(type(repr(ans)))
        # print(ans==object("1"))
        # keep = input("계속 하시겠습니까(y,n)? : ")
        # if keep == 'n':
        if repr(ans) == "0":
            print('done')
            break


def precision(pred, label):  # 1이라고 예측 한것 중 실제 1인 것
    FP = 0
    TP = 0
    for i in range(len(pred)):
        if (pred[i][0] == 1):
            if (label[i][0] == 1):
                TP += 1
            elif (label[i][0] == 0):
                FP += 1

    return (TP / (TP + FP))

def recall(pred, label):  # 실제 1인것 중 1이라고 예측한 것
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(pred)):
        if (pred[i][0] == 1):
            if (label[i][0] == 1):
                TP += 1
            elif (label[i][0] == 0):
                FP += 1
        elif (pred[i][0] == 0):
            if (label[i][0] == 1):
                FN += 1
            elif (label[i][0] == 0):
                TN += 1

    return (TP / (TP + FN))


if __name__ == "__main__":
   main()