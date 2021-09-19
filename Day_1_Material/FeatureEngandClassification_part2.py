# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 19:02:04 2021

@author: Abdul Qayyum
"""

#%% using FAST,SURF,SIFT, and HOG descripter as a features for machine learning
# classification
#

import cv2
import numpy as np
import os

from skimage.filters.rank import entropy
from skimage.morphology import disk
# Get the training classes names and store them in a list
#Here we use folder names for class names
import pandas as pd
#train_path = 'dataset/train'  # Names are class1,class2,class3 etc
train_path = 'C:\\Users\\Administrateur\\Desktop\\ENIB_work\\workshop_Isblue\\dataset'  
train_folder= os.listdir(train_path)
imagepathclass=[]
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage 
#####################################
#SIFT and SURF - do not work in opencv 3
#SIFT stands for scale invariant feature transform
#####################################
# FAST
# Features from Accelerated Segment Test
# High speed corner detector
# FAST is only keypoint detector. Cannot get any descriptors. 
import cv2 as cv
pathimage="C:\\Users\\Administrateur\\Desktop\\ENIB_work\\workshop_material\\Day1_matrial\\train\\class3Max\\14.png"
#load image
import cv2

img = cv2.imread(pathimage, 0)

# Initiate FAST object with default values
detector = cv2.FastFeatureDetector_create(50)   #Detects 50 points

kp = detector.detect(img, None)

img2_fast = cv2.drawKeypoints(img, kp, None, flags=0)
img2_fast=img2_fast[:,:,1]
fast_img = img2_fast.reshape(-1)


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

orb = cv2.ORB_create(100)


kp, des = orb.detectAndCompute(img, None)

# draw only keypoints location,not size and orientation
#img2 = cv2.drawKeypoints(img, kp, None, flags=None)
# Now, let us draw with rich key points, reflecting descriptors. 
# Descriptors here show both the scale and the orientation of the keypoint.
img2 = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_orb=img2[:,:,1]
orb_img = img2_orb.reshape(-1)

#LBP
import matplotlib.pyplot as plt
import skimage.feature
import skimage.segmentation
img_ku = skimage.feature.local_binary_pattern(img,8,1.0,method='default')
img_ku = img_ku.astype(np.uint8)
img_lbp = img_ku.reshape(-1)
#HOG
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure


fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
hog_img = hog_image_rescaled.reshape(-1)


#Dense DAISY feature description
#https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_daisy.html#sphx-glr-auto-examples-features-detection-plot-daisy-py
from skimage.feature import daisy
from skimage import data
import matplotlib.pyplot as plt

descs, descs_img = daisy(img, step=180, radius=58, rings=2, histograms=6,
                         orientations=8, visualize=True)

descs_img1=descs_img[:,:,1]
descs_img2 = descs_img1.reshape(-1)


def Descriptor_Features(img):
    df = pd.DataFrame()
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = img.reshape(-1)
    #img2 = np.mean(img)
    df['Original Image'] = img2
    df['ORB'] = orb_img 
    #gaussian_img1 = np.mean(gaussian_img)
    df['FAST'] = fast_img
    #edges1=np.mean(edges)
    df['LBP'] = img_lbp #Add column to original dataframe
    df['HOG'] = hog_img
    df['DAISY feature'] = descs_img2
    #CANNY EDGE
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    #edges1=np.mean(edges)
    df['Canny Edge'] = edges1 #Add column to original dataframe

    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    #edge_roberts1 = np.mean(edge_roberts)
    df['Roberts'] = edge_roberts1

    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    #edge_sobel1 = np.mean(edge_sobel)
    df['Sobel'] = edge_sobel1

    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    #edge_scharr1 = np.mean(edge_scharr)
    df['Scharr'] = edge_scharr1

    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    #edge_prewitt1 = np.mean(edge_prewitt)
    df['Prewitt'] = edge_prewitt1

    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    #gaussian_img1 = np.mean(gaussian_img)
    df['Gaussian s3'] = gaussian_img1

    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    #gaussian_img3 = np.mean(gaussian_img2)
    df['Gaussian s7'] = gaussian_img3

    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    #median_img1 = np.mean(median_img)
    df['Median s3'] = median_img1

    #VARIANCE with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    #variance_img1 = np.mean(variance_img)
    df['Variance s3'] = variance_img1  #Add column to original dataframe
    #Feature1 is our original image pixels
    return df

class_id=0
image_classes=[]
fea=[]
for ii in train_folder: # we have three classes folders(class1,class2,class3)
    print(ii)
    pathf=os.path.join(train_path,ii)
    print(pathf)
    imdir=os.listdir(pathf)
    print(imdir)
    for jj in imdir:  # from each class folder extract the images path
        print(jj)
        imgpath=os.path.join(pathf,jj)
        print(imgpath)
        imagepathclass.append(imgpath) # append images path as a list
        df=Descriptor_Features(imgpath) # extract features 
        df1=np.mean(df) # take mean for eevey pixel feature
        fea.append(df1) # append features in a list
        
    image_classes+=[class_id]*len(imdir)  # save class labels for each class
    class_id+=1
im_features=np.array(fea) # feature matrix

# Feature normalization in sklearn to normalize feature
from sklearn.preprocessing import StandardScaler
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)
X=im_features
Y=image_classes

#split the data into training and testing.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

################ define classifier ################
############ you can try bunch of algorithms from scikit-learn
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 10 decision trees
model = RandomForestClassifier(n_estimators = 10, random_state = 30)
################# Train the model on training data
model.fit(X_train, y_train)
##################### predict the model using test dataset #############
prediction_test = model.predict(X_test)
#print(y_test, prediction_test)
################################# check accuracy score based on prediction ########
from sklearn import metrics
#Print the prediction accuracy
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))