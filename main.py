import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import transform

palawan = imread('palawan.jpg')
# imshow(palawan)
# plt.show()

area_of_interest = [(500, 600),
                    (3220, 1950),
                    (3220, 2435),
                    (500, 3100)]
area_of_projection = [(100, 1000),
                      (3400, 1000),
                      (3400, 2600),
                      (100, 2600)]


def project_planes(image, src, dst):
    x_src = [val[0] for val in src] + [src[0][0]]
    y_src = [val[1] for val in src] + [src[0][1]]
    x_dst = [val[0] for val in dst] + [dst[0][0]]
    y_dst = [val[1] for val in dst] + [dst[0][1]]

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))

    new_image = image.copy()
    projection = np.zeros_like(new_image)
    ax[0].imshow(new_image)
    ax[0].plot(x_src, y_src, 'r--')
    ax[0].set_title('Area of Interest')

    ax[1].imshow(projection)
    ax[1].plot(x_dst, y_dst, 'r--')
    ax[1].set_title('Area of Projection')
    plt.show()


# project_planes(palawan, area_of_interest, area_of_projection)


def project_transform(image, src, dst):
    x_dst = [val[0] for val in dst] + [dst[0][0]]
    y_dst = [val[1] for val in dst] + [dst[0][1]]

    tform = transform.estimate_transform('projective',
                                         np.array(src),
                                         np.array(dst))
    transformed = transform.warp(image, tform.inverse)

    plt.figure(figsize=(6, 6))
    plt.imshow(transformed)
    plt.plot(x_dst, y_dst, 'r--')
    plt.show()


project_transform(palawan, area_of_interest, area_of_projection)