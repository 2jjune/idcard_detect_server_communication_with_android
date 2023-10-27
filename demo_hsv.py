import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf

from keras.applications.resnet_v2 import ResNet50V2
from keras import layers
from keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Activation, Dropout
from tensorflow.keras.regularizers import l2
from keras.models import Model
import demo

demo.main()

#이미지 저장 폴더
save_path = 'C:/Users/USER/PycharmProjects/homography/segment_result/demo_frame/'

def make_np_array(save_path):
    # 이미지 이름 리스트
    image_name_list = []
    #이미지 숫자 리스트
    img_num_list = []
    image_name_list = os.listdir(save_path)
    img_num_list = list(range(len(image_name_list)))
    np_list = []
    while(len(img_num_list)>5):
        #이미지 숫자 리스트에서 램덤으로 하나 숫자 꺼내오기
        r1 = img_num_list.pop(random.randrange(0,len(img_num_list)))
        r2 = img_num_list.pop(random.randrange(0,len(img_num_list)))
        r3 = img_num_list.pop(random.randrange(0,len(img_num_list)))
        r4 = img_num_list.pop(random.randrange(0,len(img_num_list)))
        r5 = img_num_list.pop(random.randrange(0,len(img_num_list)))
        #램덤 이미지 선택
        img = cv2.imread(save_path+image_name_list[r1])
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = cv2.imread(save_path+image_name_list[r2])
            # img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(save_path+image_name_list[r3])
            # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img3 = cv2.imread(save_path+image_name_list[r4])
            # img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img4 = cv2.imread(save_path+image_name_list[r5])
            # img4 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = np.concatenate((img,img1,img2,img3,img4), axis=2)
        np_list.append(x)
    return np_list


def image_generator_rgb(np_file):
    img = []
    x1 = np.copy(np_file)
    for i in range(5):
        x1[:,:,3*(i):3*(i+1)] = cv2.cvtColor(x1[:,:,3*(i):3*(i+1)],cv2.COLOR_BGR2HSV)
    for i in range(x1.shape[2]):
        img.append(np.reshape((x1[:,:,i]),(512,512,1)))
    img = np.array(img)
    x1 = np.concatenate((img[0],img[1],img[2],img[3],img[4],img[5],img[6],img[7],img[8],img[9],
    img[10],img[11],img[12],img[13],img[14]), axis=2)

    x1 = np.float32(x1)
    x_r = x1[:,:,0::3]
    x_g = x1[:,:,1::3]
    x_b = x1[:,:,2::3]

    # print(x_v)
    x_r_mean = np.mean(x_r,axis=2)
    x_g_mean = np.mean(x_g, axis=2)
    x_b_mean = np.mean(x_b, axis=2)

    x_r_std = np.std(x_r,axis=2)
    x_g_std = np.std(x_g, axis=2)
    x_b_std = np.std(x_b, axis=2)

    image_mean = cv2.merge((x_r_mean, x_g_mean, x_b_mean))
    image_std = cv2.merge((x_r_std, x_g_std, x_b_std))

    image = np.concatenate((image_mean,image_std), axis=2)
    image = np.float32(image)
    image[:,:,0] = image[:,:,0]/255.
    image[:,:,3] = image[:,:,3]/255.

    image[:,:,1] = image[:,:,1]/255.
    image[:,:,4] = image[:,:,4]/255.

    image[:,:,2] = image[:,:,2]/255.
    image[:,:,5] = image[:,:,5]/255.

    return image[:,:,3:6]

threshold = 0.5

# theshold 값조정
def encoding_theshold(pred):
  for i in range(len(pred)):
    if pred[i] >=threshold:
      pred[i] = 1
    elif pred[i]<threshold:
      pred[i] = 0
  return pred


def precision(pred, label): #1이라고 예측 한것 중 실제 1인 것
  FP = 0
  TP = 0
  for i in range(len(pred)):
    if (pred[i][0] == 1):
      if (label[i][0]==1):
        TP +=1
      elif (label[i][0]==0):
        FP +=1

  return (TP/(TP+FP))

def recall(pred,label): #실제 1인것 중 1이라고 예측한 것
  TP =0
  FP = 0
  TN = 0
  FN = 0

  for i in range(len(pred)):
    if(pred[i][0]==1):
      if(label[i][0] == 1):
        TP +=1
      elif (label[i][0] == 0):
        FP +=1

    elif(pred[i][0]==0):
      if(label[i][0] == 1):
        FN +=1
      elif (label[i][0] == 0):
        TN +=1


  return (TP/(TP+FN))


img_height,img_width,input_channel = 512,512,3
resnet50 = ResNet50V2(include_top=False, input_shape=(512,512,3))

x = resnet50.output
x = MaxPooling2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(resnet50.input, outputs = x)
callbacks = [
             tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
             tf.keras.callbacks.ModelCheckpoint('checkpoint/adagrad_decay_5e-2.h5', monitor = 'val_loss', save_weights_only=True,save_best_only=True)
]
model.summary()
model.compile(tf.keras.optimizers.Adagrad(),
                loss='binary_crossentropy',
              metrics=['accuracy'])
model.load_weights('./hsv.h5')
np_list = make_np_array(save_path)

for i in range(len(np_list)):
    img = image_generator_rgb(np_list[i])
    img = np.reshape(img,(1,512,512,3))
    pred = model.predict(img)
    if pred >= 0.5:
        print('진짜 신분증')
    else:
        print("가짜 신분증")
