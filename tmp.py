import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
import PIL
import random
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50
from keras import layers
from keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Activation, Dropout
from tensorflow.keras.regularizers import l2
from keras.models import Model
from focal_loss import BinaryFocalLoss



# Train ImageDataGenerator
train_gen = ImageDataGenerator(rescale=1/255.)
# Test ImageDataGenerator
test_gen = ImageDataGenerator(rescale=1/255.)


train_flow_gen = train_gen.flow_from_directory(directory='C:/Users/USER/PycharmProjects/homography/data(1image)/train',
                                              target_size=(512, 512),  # 사용할 CNN 모델 입력 사이즈에 맞게 resize
                                              class_mode='binary',
                                              batch_size=32,
                                              shuffle=True)

test_flow_gen = test_gen.flow_from_directory(directory='C:/Users/USER/PycharmProjects/homography/data(1image)/test',
                                            target_size=(512, 512),  # 사용할 CNN 모델 입력 사이즈에 맞게 resize
                                            class_mode='binary',
                                            batch_size=32,
                                            shuffle=False)



# resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(512,512,3))
resnet = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=(512,512,3))
vgg = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=(512,512,3))

x = resnet.output
x = Flatten()(x)
# x = Dense(5000, activation='relu')(x)
# x = Dense(4000, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(resnet.input, outputs = x)
# model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
#                                input_shape=(720, 720, 3)),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#         # tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dropout(0.21),
#         tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         # tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         # tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dropout(0.35),
#         tf.keras.layers.Conv2D(256, kernel_size=(2, 2), activation='relu', padding='same'),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         # tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dropout(0.38),
#         tf.keras.layers.Conv2D(512, kernel_size=(2, 2), activation='relu', padding='same'),
#         # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dropout(0.4),
#
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(1000, activation='relu'),
#         # tf.keras.layers.Conv2DTranspose(3,3,strides=2, padding='same')
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])





model.summary()



# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.Adagrad(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
history = model.fit(train_flow_gen, epochs=100, shuffle=True, validation_data = (test_flow_gen))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 7 훈련 과정 시각화 (손실)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# result = model.predict(img)

print(result)