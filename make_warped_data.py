import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
coodinates = {}

width = 512
height = 512

file_path = 'sample_jyj.json'
with open(file_path, 'r') as file:
    data = json.load(file)
    # print(type(data))

for i in range(len(data)):
    if data[i]['image'] in coodinates.keys():
        coodinates[data[i]['image']].append(data[i]['coordinate'][0])
    else:
        coodinates[data[i]['image']] = data[i]['coordinate']

# print(data)
# print(coodinates)

file_path = 'sample_hic.json'
with open(file_path, 'r') as file:
    data = json.load(file)
    # print(type(data))

for i in range(len(data)):
    if data[i]['image'] in coodinates.keys():
        coodinates[data[i]['image']].append(data[i]['coordinate'][0])
    else:
        coodinates[data[i]['image']] = data[i]['coordinate']

file_path = 'sample.json'
with open(file_path, 'r') as file:
    data = json.load(file)
    # print(type(data))

for i in range(len(data)):
    if data[i]['image'] in coodinates.keys():
        coodinates[data[i]['image']].append(data[i]['coordinate'][0])
    else:
        coodinates[data[i]['image']] = data[i]['coordinate']

file_path = 'sample1.json'
with open(file_path, 'r') as file:
    data = json.load(file)
    # print(type(data))

for i in range(len(data)):
    if data[i]['image'] in coodinates.keys():
        coodinates[data[i]['image']].append(data[i]['coordinate'][0])
    else:
        coodinates[data[i]['image']] = data[i]['coordinate']

file_path = 'sample2.json'
with open(file_path, 'r') as file:
    data = json.load(file)
    # print(type(data))

for i in range(len(data)):
    if data[i]['image'] in coodinates.keys():
        coodinates[data[i]['image']].append(data[i]['coordinate'][0])
    else:
        coodinates[data[i]['image']] = data[i]['coordinate']

# for key, value in coodinates.items():
#     image_name = key
#     value = np.array(value)
#     pts1 = np.float32(value)
#     img = cv2.imread('make_co/' + image_name)
#     pts2 = np.float32([[0, 0], [width - 1, 0],
#                        [width - 1, height - 1], [0, height - 1]])
#     mtrx = cv2.getPerspectiveTransform(pts1, pts2)
#     result = cv2.warpPerspective(img, mtrx, (width, height))
#     cv2.imshow('scanned', result)
#     cv2.waitKey()



for key, value in coodinates.items():
    image_name = key
    value = np.array(value)
    pts1 = np.float32(value)
    img = cv2.imread('make_co/' + image_name)
    pts2 = np.float32([[0, 0], [width - 1, 0],
                       [width - 1, height - 1], [0, height - 1]])
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, mtrx, (width, height))

    # masked = padding(result,1024)
    masked = cv2.copyMakeBorder(result, 256, 256, 256, 256, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    print(masked.shape)
    # plt.imshow(masked)
    # plt.show()
    # cv2.imshow('scanned', masked)
    # cv2.waitKey()
    cv2.imwrite('./warped_padding/{}'.format(image_name), masked)