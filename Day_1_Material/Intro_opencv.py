# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 08:41:04 2021

@author: Abdul Qayyum
ENIB
"""

#Introduction of OpenCv for basic Image processing

#Useful preprocessing steps for image processing, for example segmentation. 
#1. Reading and processing BGR and GRAY scale channels
#1. channels extraction(SPlit & Merge)
#2. Scaling / resizing
#3. Denoising / smoothing
#4. Edge detection


###################################
#Pixel values, split and merge channels, 
#pip install opencv-contrib-python==4.5.3.56
import cv2
# check open cv version
cv2.__version__

# Read gray and BGR or RGB image in OpenCV
grayim=cv2.imread("Test_images\\Desmodora sp1_GY_8S1D1 (13).jpg",0)

img_bgr=cv2.imread("Test_images\\Desmodora sp1_GY_8S1D1 (13).jpg",1) # color is BGR not RGB
############## BGR to RGB color image
img_bgr_rgb= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# another way to get gray  image from BGR color image
gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
######################## back to RGB color
# gray to RGB back 
backtobgr = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2BGR)
backtorgb = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2RGB)
# another way to back RGB
gray_three = cv2.merge([gray_img,gray_img,gray_img])

# show individual channel in color image
img_b=img_bgr[:,:,0] # B dueto BGR
img_g=img_bgr[:,:,1] # green
img_r=img_bgr[:,:,2] # red
import matplotlib.pyplot as plt
plt.imshow(img_r)
# show image in openCv
# cv2.imshow("red pic", img_r)
# cv2.waitKey(0)          
# cv2.destroyAllWindows() 

# split channels
b,g,r=cv2.split(img_bgr)

# cv2.imshow("green pic", g)
# cv2.waitKey(0)          
# cv2.destroyAllWindows() 

#back to combine into bgr


img_merged = cv2.merge((b,g,r))


# cv2.imshow("merged pic", img_merged)
# cv2.waitKey(0)          
# cv2.destroyAllWindows() 

## Resize the image according to your specified size
img = cv2.imread("Test_images\\Desmodora sp1_GY_8S1D1 (13).jpg", 1)   #Color is BGR not RGB

#use cv2.resize. Can specify size or scaling factor.
#Inter_cubic or Inter_linear for zooming.
#Use INTER_AREA for shrinking
#Following xample zooms by 2 times.

resized = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

cv2.imshow("original pic", img)
cv2.imshow("resized pic", resized)
cv2.waitKey(0)          
cv2.destroyAllWindows()


# resized_s = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_AREA)

# cv2.imshow("original pic", img)
# cv2.imshow("resized pic", resized_s)
# cv2.waitKey(0)          
# cv2.destroyAllWindows()  

######### downscale resize
height=128
width=128

resized_1 = cv2.resize(img,(height,width))

cv2.imshow("original pic", img)
cv2.imshow("resized pic", resized_1)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

# another way
height=128
width=128

resized_1 = cv2.resize(img,(height,width),interpolation = cv2.INTER_CUBIC)

cv2.imshow("original pic", img)
cv2.imshow("resized pic", resized_1)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

########### cropped image
cropped_img=img[100:200,100:200]

cv2.imshow("original pic", img)
cv2.imshow("cropped pic", cropped_img)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

################## some basic filtering operations in OpenCV #################

import cv2
import numpy as np
from matplotlib import pyplot as plt
img_bgr= cv2.imread('Test_images\\Desmodora sp1_GY_8S1D1 (13).jpg', 1)
kernel = np.ones((5,5),np.float32)/25
filt_2D = cv2.filter2D(img_bgr,-1,kernel)    #Convolution using the kernel we provide
blur = cv2.blur(img_bgr,(5,5))   #Convolution with a normalized filter.
blur_gaussian = cv2.GaussianBlur(img,(5,5),0)  #Gaussian kernel is used. 
median_blur = cv2.medianBlur(img_bgr,5)  #Using kernel size 5. Better on edges compared to gaussian.
bilateral_blur = cv2.bilateralFilter(img_bgr,9,75,75)  #Good for noise removal but retain edge sharpness. 


cv2.imshow("Original", img_bgr)
cv2.imshow("2D filtered", filt_2D)
cv2.imshow("Blur", blur)
cv2.imshow("Gaussian Blur", blur_gaussian)
cv2.imshow("Median Blur", median_blur)
cv2.imshow("Bilateral", bilateral_blur)
cv2.waitKey(0)          
cv2.destroyAllWindows() 


import cv2
# Sobel Edge Detection
sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis

sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis

sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

# Display Sobel Edge Detection Images

cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)


#Edge detection  
    
import cv2
import numpy as np

img = cv2.imread("Test_images\\Desmodora sp1_GY_8S1D1 (13).jpg", 0)
edges = cv2.Canny(img,100,200)   #Image, min and max values

cv2.imshow("Original Image", img)
cv2.imshow("Canny", edges)

cv2.waitKey(0)          
cv2.destroyAllWindows() 
############################## Feature descriptor and dectector #########
######################### feature descriptor ##########
#SIFT and SURF - do not work in opencv 3
#SIFT stands for scale invariant feature transform
#####################################
# FAST
# Features from Accelerated Segment Test
# High speed corner detector
# FAST is only keypoint detector. Cannot get any descriptors. 
import cv2 as cv
pathimage="Test_images\\Desmodora sp1_GY_8S1D1 (13).jpg"
#load image
import cv2

img = cv2.imread(pathimage, 0)

# Initiate FAST object with default values
detector = cv2.FastFeatureDetector_create(50)   #Detects 50 points

kp = detector.detect(img, None)

img2 = cv2.drawKeypoints(img, kp, None, flags=0)
fast_img = img2.reshape(-1)
cv2.imshow('Corners',img2)
cv2.waitKey(0)

#############################################
#BRIEF (Binary Robust Independent Elementary Features)
#One important point is that BRIEF is a feature descriptor, 
#it doesn’t provide any method to find the features.
# Not going to show the example as BRIEF also not working in opencv 3

###############################################
#ORB
# Oriented FAST and Rotated BRIEF
# An efficient alternative to SIFT or SURF
# ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor

import numpy as np
import cv2

img = cv2.imread(pathimage, 0)
#Initiate SIFT detector
orb = cv2.ORB_create(100)


kp, des = orb.detectAndCompute(img, None)

# draw only keypoints location,not size and orientation
#img2 = cv2.drawKeypoints(img, kp, None, flags=None)
# Now, let us draw with rich key points, reflecting descriptors. 
# Descriptors here show both the scale and the orientation of the keypoint.
img2 = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
orb_img = img2.reshape(-1)
cv2.imshow("With keypoints", img2)
cv2.waitKey(0)


# SIFT and Surf not in openCV current virsion
# import cv2
# import numpy as np

# img = cv2.imread('home.jpg')
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sift = cv2.SIFT()
# kp = sift.detect(img,None)

# img=cv2.drawKeypoints(img,kp)

# cv2.imwrite('sift_keypoints.jpg',img)



# >>> img = cv2.imread('fly.png',0)

# # Create SURF object. You can specify params here or later.
# # Here I set Hessian Threshold to 400
# >>> surf = cv2.SURF(400)

# # Find keypoints and descriptors directly
# >>> kp, des = surf.detectAndCompute(img,None)

# >>> len(kp)
#  699

########################### some other function in OpenCV ##############
#Morphological transformations
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("Test_images\\Desmodora sp1_GY_8S1D1 (13).jpg", 0)

ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)   # 3x3 kernel with all ones. 
erosion = cv2.erode(th,kernel,iterations = 1)  #Erodes pixels based on the kernel defined

dilation = cv2.dilate(erosion,kernel,iterations = 1)  #Apply dilation after erosion to see the effect. 

#Erosion followed by dilation can be a single operation called opening
opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)  # Compare this image with the previous one

#Closing is opposit, dilation followed by erosion.
closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

#Morphological gradient. This is the difference between dilation and erosion of an image
gradient = cv2.morphologyEx(th, cv2.MORPH_GRADIENT, kernel)

#It is the difference between input image and Opening of the image. 
tophat = cv2.morphologyEx(th, cv2.MORPH_TOPHAT, kernel)

#It is the difference between the closing of the input image and input image.
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow("Original Image", blackhat)
cv2.waitKey(0)          
cv2.destroyAllWindows() 












