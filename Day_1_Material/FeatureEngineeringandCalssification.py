# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 19:00:27 2021

@author: Abdul Qayyum
"""

#%% Classification based on handcrafted(filtering based features) using
#Machine learning
import cv2
import numpy as np
import os

from skimage.filters.rank import entropy
from skimage.morphology import disk
# Get the training classes names and store them in a list
#Here we use folder names for class names
import pandas as pd
#train_path = 'dataset/train'  # Names are class1,class2,class3 etc
train_path = 'C:\\Users\\Administrateur\\Desktop\\ENIB_work\\workshop_Isblue\\dataset'  #
train_folder= os.listdir(train_path)
imagepathclass=[]
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage 
######################## develop feature matrix ####################
df = pd.DataFrame()
def featureextraction(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = img.reshape(-1)
    #img2 = np.mean(img)
    df['Original Image'] = img2
    entropy_img = entropy(img, disk(1))
    entropy1 = entropy_img.reshape(-1)
    #entropy1 = np.mean(entropy_img)
    df['Entropy'] = entropy1
    #df1=df.mean(axis=0)
    gaussian_img = ndimage.gaussian_filter(img, sigma=3)

    gaussian_img1 = gaussian_img.reshape(-1)
    #gaussian_img1 = np.mean(gaussian_img)
    df['Gaussian s3'] = gaussian_img1


    #Gerate OTHER FEATURES and add them to the data frame
                
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
    #Generate Gabor features
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
#               print(theta, sigma, , lamda, frequency)
                
                    gabor_label = 'Gabor' + str(num)
#                    print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter image and add values to new column
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Modify this to add new column for each gabor
                    num += 1
    
    
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
        df=featureextraction(imgpath) # extract features 
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