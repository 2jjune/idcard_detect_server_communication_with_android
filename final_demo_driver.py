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
import time
import matplotlib.pyplot as plt
import gc

class check_finish:
   q = 0
   def __init__(self):
      global win
      btn = Button(win, text="종료", command=self.check_finish_yes, width=7, height=2)
      btn.pack(side='bottom')
      btn.pack(pady=40)


   def __repr__(self):
      return "% s" % (self.q)

   def check_finish_yes(self):
      self.q = 1
      win.destroy()
      win.update()
      return self.q

class check_finish_result:
    tmp = 0
    global result_win
    def __init__(self):
        # result_win = tk.Toplevel(win)
        label=tk.Label(result_win, text='계속 진행하시겠습니까?', font=35)
        label.place(x=587, y=525)


        btn = Button(result_win, text="예", command=self.check_continue_yes, width=5)
        btn2 = Button(result_win, text="아니오", command=close_result_app, width=5)
        btn.place(x=600, y=552)
        btn2.place(x=700, y=552)

        result_win.mainloop()

    def __repr__(self):
        return "% s" % (self.tmp)

    def check_continue_yes(self):
        self.tmp = 1
        result_win.destroy()
        result_win.update()
        # window.destroy()
        return self.tmp

def delete_files(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)

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

def close_app():
   global win
   win.destroy()
   win.update()

def close_toplevel_app():
    global new_window
    new_window.destroy()
    new_window.update()

def check_5pic():
    global win, new_window, save_path
    # if len(os.listdir('C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame')) < 5:
    if len(os.listdir(save_path)) < 5:
        new_window = tk.Toplevel(win)
        # pic_window.update()
        # new_win = tk.Toplevel(cam_window)
        label = tk.Label(new_window, text='사진을 더 찍으세요', font=font.Font(size=20, weight='bold'), fg='red')
        label.pack(side='top')
        label.pack(pady=5)
        label = tk.Label(new_window, text='워핑된 사진 : {}'.format(
            len(os.listdir(save_path))), font=35)
            # len(os.listdir('C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame'))), font = 35)
        label.pack(side='top')
        label.pack(pady=5)
        label = tk.Button(new_window, text='확인', font=font.Font(size=15, weight='bold', underline=1), command=close_toplevel_app)
        label.pack(side='bottom')
        label.pack(pady=5)
        # pic_window.mainloop()

        print('사진을 더 찍으세요')
    else:
        win.destroy()
        win.update()

def screen_shot():
   global num, warp_model, img_size, save_path

   cv2image= cap.read()[1]
   cv2image = cv2image[:, int(cv2image.shape[1] * 0.125):int(cv2image.shape[1] * 0.875)]
   # cv2.imwrite(f"C:/Users/USER/PycharmProjects/homography/segment_result/webcam/capture_{num}.jpg", cv2image)


   #워핑 하는 부분----------------------------------------------------------------

   image = cv2.resize(cv2image, (img_size, img_size))
   image = np.asarray(image)
   image = image / 255.
   image = np.expand_dims(image, axis=0)

   pred_mask = warp_model.predict(image)

   pred_mask = np.squeeze(pred_mask, axis=0)
   pred_mask2 = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 1))

   for i in range(pred_mask.shape[0]):
       for j in range(pred_mask.shape[1]):
           if pred_mask[i][j] > 0.5:
               pred_mask2[i][j] = 255
   save_img = tf.keras.utils.array_to_img(pred_mask2)

   # save_img.save('./segment_result/{0}_{1}.jpg'.format(file[:-4],check_num))

   save_img = np.array(save_img)
   save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)

   receipt_image = make_scan_image(cv2image, save_img, min_threshold=25, max_threshold=70)

   if type(receipt_image) == int:
       pass
   else:
       # cv2.imwrite(f'C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame/capture_{num}.jpg',receipt_image)
       cv2.imwrite(save_path+f'capture_{num}.jpg',receipt_image)
   del pred_mask, pred_mask2, save_img, receipt_image
   num += 1

# Define function to show frame
def show_frames():
   # Get the latest frame and convert into Image
   global  cv2image, cap, label, save_path
   # num_of_file = "pic:" + str(len(os.listdir('C:/Users/USER/PycharmProjects/homography/segment_result/webcam/')))
   # num_of_warp_file = "warp:" + str(len(os.listdir('C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame/')))

   num_of_warp_file = "warp:" + str(len(os.listdir(save_path)))

   # cap_img = cv2.putText(cap.read()[1], num_of_file, (1550, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=8)
   # cap_img = cv2.putText(cap_img, num_of_warp_file, (1000, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=8)

   # cap_img = cv2.putText(cap.read()[1], num_of_warp_file, (1550, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=8)
   cap_img = cv2.putText(cap.read()[1], num_of_warp_file, (int(cap.read()[1].shape[1]*0.8), int(cap.read()[1].shape[0]*0.12)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=8)
   cv2image= cv2.cvtColor(cap_img,cv2.COLOR_BGR2RGB)

   small = cv2.resize(cv2image, (1024, 576))
   img = Image.fromarray(small)
   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)
   # Repeat after an interval to capture continiously
   label.after(20, show_frames)


def delete_files(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)

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

def close_result_app():
   global result_win
   result_win.destroy()
   result_win.update()

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

def print_result(img_list, pred):
    global win, result_win, classification_model
    result_win = Tk()
    result_win.title("결과")
    result_win.geometry("1350x600")

    img = []
    photo = []
    photo_label = []
    second_test_img = []
    for i in range(0,5):
        img.append(img_list[:,:,int(i+(i*2)):int(3+(3*i))])
        img[i] = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
        img[i] = cv2.resize(img[i], (256, 256))
        img[i] = Image.fromarray(img[i])
        photo.append(ImageTk.PhotoImage(image=img[i]))
        photo_label.append(Label(result_win, image=photo[i]))
        photo_label[i].pack(side='left')
        photo_label[i].pack(padx=5)


    if pred>0.5:
        fram_pic_dir = './demo_frame_pic/'
        if not os.path.exists(fram_pic_dir):
            os.makedirs(fram_pic_dir)
        delete_files(fram_pic_dir)

        for i in range(0, 5):
            second_test_img.append(img_list[:, :, int(i + (i * 2)):int(3 + (3 * i))])
            #######신분증##########
            # second_test_img[i][:int(512 * 0.05), :] = 255
            # second_test_img[i][int(512 * 0.735):, :] = 255
            # second_test_img[i][:, :int(512 * 0.59)] = 255
            # second_test_img[i][:, int(512 * 0.985):] = 255

            #######면허증##########
            second_test_img[i][:int(512 * 0.236), :] = 255
            second_test_img[i][int(512 * 0.928):, :] = 255
            second_test_img[i][:, :int(512 * 0.018)] = 255
            second_test_img[i][:, int(512 * 0.38):] = 255
            cv2.imwrite(fram_pic_dir+f'{i}.jpg',second_test_img[i])
        del img
        np_list = make_np_array(fram_pic_dir)
        img = image_generator_rgb(np_list[0])


        img = np.reshape(img, (1, 512, 512, 3))
        second_pred = classification_model.predict(img)
        print('second pred', second_pred)
        if second_pred>0.4:
            label = tk.Label(result_win, text=f'전체 영역 진품 (confidence={pred[0][0]*100:.2f}%)', font=font.Font(size=17, weight='bold', underline=1), background='white')
            label2 = tk.Label(result_win, text='최종 합격!', font=font.Font(size=19, weight='bold', underline=1), background='white')
            label.place(x=480, y=440)
            label2.place(x=600, y=480)
        else:
            label = tk.Label(result_win, text=f'전체 영역 진품 (confidence={pred[0][0]*100:.2f}%)', font=font.Font(size=17, weight='bold', underline=1), background='white')
            label2 = tk.Label(result_win, text='사진 영역 불합격', font=font.Font(size=19, weight='bold', underline=1), background='white')
            label.place(x=480, y=440)
            label2.place(x=570, y=480)

    else:
        label = tk.Label(result_win, text=f'위조 신분증입니다 (confidence={100-pred[0][0]*100:.2f}%)', font=font.Font(size=20, weight='bold', underline=1), background='white')
        label.place(x=445, y=460)

    ans = check_finish_result()

    del img, photo, photo_label, second_test_img
    result_win.mainloop()
    return ans

def webcam(a):
   # Create an instance of TKinter Window or frame
   global cap, label, cv2image, win, num
   num = a

   win = Tk()
   # win.update()
   # Set the size of the window
   win.geometry("1280x720")
   win.title("NEW WINDOW")

   # Create a Label to capture the Video frames
   label = Label(win)
   label.pack(side='top')

   # time.sleep(0.5)
   cap = cv2.VideoCapture(1)
   # cv2image = None

   button = tk.Button(win, text="완료", command=check_5pic, width=15, height=2, bg='white', fg='red',
                      overrelief='solid', font=font.Font(size=13, weight='bold', underline=1))
   button2 = tk.Button(win, text="촬영", command=screen_shot, width=15, height=2, bg='white', overrelief='solid', font=font.Font(size=13, weight='bold', underline=1))
   button.pack(side='left')
   button2.pack(side='right')
   button.pack(padx=200)
   button2.pack(padx=200)

   show_frames()

   q = check_finish()
   win.mainloop()

   del cap

   return (num,q)

def warp_main(model):
    a = 0
    global warp_model, img_size, save_path
    img_size = 512
    warp_model = model
    while(1):
        # webcam_Path = 'C:/Users/USER/PycharmProjects/homography/segment_result/webcam/'
        # webcam_Path = './webcam/'
        # if not os.path.exists(webcam_Path):
        #     os.makedirs(webcam_Path)
        # delete_files(webcam_Path)

        a,q = webcam(a)

        # webcam_img_list = os.listdir(webcam_Path)
        if repr(q)=="1":
            del q
            return 'q'

        # if len(os.listdir('C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame'))>=5:
        if len(os.listdir(save_path)) >= 5:
            break

    del a

def main():

    now = datetime.now()
    max_date = datetime.strptime("202311091105","%Y%m%d%H%M")
    date_diff = now-max_date

    if date_diff.days>=0:
        return

    global save_path, classification_model
    # save_path = 'C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame/'
    save_path = './demo_frame/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    warp_model = build_resnet50_unet((512, 512, 3))
    warp_model.load_weights('./best_with_RESNET50_512_focalloss_13200_data_shuffle.h5')

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

    while(True):
        now = datetime.now()
        date_diff = now - max_date
        if date_diff.days >= 0:
            return

        # delete_files('C:/Users/USER/PycharmProjects/homography/segment_result/webcam/')
        # delete_files('C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame/')
        delete_files(save_path)

        ##### warping part###
        q = warp_main(warp_model)
        if q=='q':
            break

        ######## classification part ######
        np_list = make_np_array(save_path)
        img = image_generator_rgb(np_list[0])
        img = np.reshape(img,(1,512,512,3))

        pred = classification_model.predict(img)
        print(pred)

        ans = print_result(np_list[0], pred)

        if repr(ans) == "0":
            print('done')
            break

        del ans, pred, img, np_list, q
        gc.collect()

if __name__ == "__main__":
   main()