import matplotlib.pyplot as plt
import cv2
import requests
import numpy as np
import os

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
# from tensorflow.keras.prerprocessing.image import ImageDataGenerator
# from tensorflow import keras
import edge_detection_model
# def display(display_list):
#     plt.figure(figsize=(15, 15))
#     title = ["Input Image", "True Mask", "Predicted Mask"]
#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i+1)
#         plt.title(title[i])
#         plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
#         plt.axis("off")
#     plt.show()
#
# def create_mask(pred_mask):
#     pred_mask = tf.argmax(pred_mask, axis=-1)
#     pred_mask = pred_mask[..., tf.newaxis]
#     return pred_mask[0]
#
# def show_predictions(dataset=None, history='./checkpoint/best_first.h5', num=1):
#     if dataset:
#         for image, mask in dataset.take(num):
#             pred_mask = history.predict(image)
#             display([image[0], mask[0], create_mask(pred_mask)])
org_image = 'test_img13.jpg'
img_height = 512
img_width = 512

# base_model = tf.keras.applications.vgg16.VGG16(
#     include_top=False, input_shape=(img_width, img_height, 3))
#
# layer_names = [
#     'block1_pool',
#     'block2_pool',
#     'block3_pool',
#     'block4_pool',
#     'block5_pool',
# ]
# base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
# base_model.trainable = False
# VGG_16 = tf.keras.models.Model(base_model.input, base_model_outputs)


# model = edge_detection_model.UNet()
# model = edge_detection_model.UNetCompiled_skip_small(input_size=(img_height,img_width,3), n_filters=64, n_classes=1)
# model = edge_detection_model.segmentation_model(img_width,img_height,VGG_16)
# model = edge_detection_model.segmentation_model(img_width,img_height,RESNET50)
model = edge_detection_model.build_resnet50_unet((img_width,img_height,3))

model.load_weights('./checkpoint/best_with_RESNET50_512_focalloss_8918_data_shuffle.h5')

# show_predictions(dataset='test_img.jpg', history=model)
image = cv2.imread(org_image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

image = cv2.resize(image, (img_height, img_width))
image = np.asarray(image)
image = image/255.
image = np.expand_dims(image, axis=0)


pred_mask = model.predict(image)


def rgb_to_gray(img):
    grayImage = np.zeros((img.shape[0], img.shape[1], 1))
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    Avg = (R + G + B)
    # grayImage = img.copy()

    for i in range(1):
        grayImage[:, :, i] = Avg

    return grayImage

print(pred_mask.shape)
print(pred_mask)
# print(max(pred_mask))
image = np.squeeze(image, axis=0)
pred_mask = np.squeeze(pred_mask, axis=0)
# pred_mask = rgb_to_gray(pred_mask)
print(pred_mask.shape)
print(pred_mask[150][150])
# print(pred_mask[0][243][333])
pred_mask2 = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 1))
for i in range(pred_mask.shape[0]):
    for j in range(pred_mask.shape[1]):
        # print(pred_mask[i][j])
        if pred_mask[i][j]>0.5:
            pred_mask2[i][j]=255
        else:
            pred_mask2[i][j]=0

title = ['input', 'predict']

plt.subplot(1, 3, 1)
plt.title('org_img')
# plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(tf.keras.utils.array_to_img(image))

plt.subplot(1, 3, 2)
plt.title('pred_img')
plt.imshow(tf.keras.utils.array_to_img(pred_mask), cmap='gray')

plt.subplot(1, 3, 3)
plt.title('pred_img_sharp')
plt.imshow(tf.keras.utils.array_to_img(pred_mask2), cmap='gray')

plt.axis('off')
plt.show()

