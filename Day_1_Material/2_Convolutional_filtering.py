# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:45:02 2021

@author: Abdul Qayyum
"""
# Filtering in scipy 
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import rescale
from scipy.signal import convolve2d
import cv2
# Read gray and BGR or RGB image in OpenCV

img_bgr=cv2.imread("Test_images\\Desmodora sp1_GY_8S1D1 (13).jpg",1) # color is BGR not RGB
############## BGR to RGB color image
img_bgr_rgb= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

########### define gray scale image
grayim=cv2.imread("Test_images\\Desmodora sp1_GY_8S1D1 (13).jpg",0)
plt.imshow(grayim);
kernel= np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]])
red = convolve2d(grayim, kernel, 'valid')

plt.imshow(kernel)


def rgb_convolve2d(image, kernel):
    red = convolve2d(image[:,:,0], kernel, 'valid')
    green = convolve2d(image[:,:,1], kernel, 'valid')
    blue = convolve2d(image[:,:,2], kernel, 'valid')
    return np.stack([red, green, blue], axis=2)
## check convolutional operation here
conv_im1 = rgb_convolve2d(img_bgr_rgb, kernel)

fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].imshow(kernel, cmap='gray')
ax[1].imshow(abs(conv_im1), cmap='gray');

########### different masks or filters can use for edge detection
# Edge Detection1
kernel1 = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])
# Edge Detection2
kernel2 = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])
# Bottom Sobel Filter
kernel3 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
# Top Sobel Filter
kernel4 = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
# Left Sobel Filter
kernel5 = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
# Right Sobel Filter
kernel6 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

##### just plot the images
kernels = [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6]
# kernel_name = ['Edge Detection’, ‘Edge Detection’, 
#                'Bottom Sobel’, ‘Top Sobel’, 
#                'Left Sobel’, ‘Right Sobel’]
               
kernel_name = ['Edge Detection', 'Edge Detection', 
               'Bottom Sobel', 'Top Sobel', 
               'Left Sobel', 'Right Sobel']
               
figure, axis = plt.subplots(2,3, figsize=(12,10))
for kernel, name, ax in zip(kernels, kernel_name, axis.flatten()):
    conv_im1 = convolve2d(grayim, kernel[::-1, ::-1]).clip(0,1)
    ax.imshow(abs(conv_im1), cmap='gray'),ax.set_title(name)

