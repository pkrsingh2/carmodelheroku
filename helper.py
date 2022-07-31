#!/usr/bin/env python
# coding: utf-8


import pickle
import os
import numpy as np
import pandas as pd
from PIL import Image
#import cv2
import tensorflow as tf
from PIL import ImageDraw as D


# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
# from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
# from tensorflow.keras.applications.densenet import DenseNet201
# from tensorflow.keras.applications.nasnet import NASNetMobile




modelName = "ResNet50V2_FINAL"
fname = os.path.join('.',"static","%s_attrib.gl"%modelName)   

with open(fname, 'rb') as fh:
    attributes = pickle.load(fh)
weights = os.path.join('.',"static","%s_weights.h5"%modelName)
inputShape = attributes.get('inputShape')

model = attributes.get('modelFunc')(input_size=inputShape,application=ResNet50V2)
model.load_weights(weights)

tkpi = tf.keras.preprocessing.image

def predictor(image):
    
    (w,h) = Image.open(image).size
    img_original = Image.open(image)
    img = np.expand_dims(np.array(img_original.copy().resize(inputShape[1:3])),axis=0)/255.0
    
    result = model.predict(img)
    enc = attributes['labelEncoder']    

    clsProba = np.round((result[0]*100),2)

    sortPos = np.argsort(clsProba)
    df = pd.DataFrame(np.array(list(zip(enc.classes_[sortPos][0],
                               clsProba[0][sortPos][0][::-1])))[:5],
                          columns=["names","confidence"])

    pBox = np.multiply(np.append((w,h),(w,h)),result[1][0]).astype(int)
    pBox[0] = max(pBox[0],0)
    pBox[1] = max(pBox[1],0)
    pBox[2] = min(pBox[2],w)
    pBox[3] = min(pBox[3],h)
    
    # img_tmp = np.array(img_original)
    # cv2.rectangle(img_tmp,pBox[:2],pBox[2:],(0,0,255),2)
    # img_final = tkpi.array_to_img(img_tmp)

    #Alternate solution to add bounding box
    i = Image.open(image)
    draw = D.Draw(i)
    draw.rectangle([(pBox[0],pBox[1]),(pBox[2],pBox[3])], outline="red")
    #i.show()

    #return df, img_final
    return df, i


