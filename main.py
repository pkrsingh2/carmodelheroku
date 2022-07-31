#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#from tkinter.ttk import Style
import streamlit as st
import os
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_theme(style="darkgrid")
#sns.set()


from PIL import Image


# define a transfer learning network mapped to our targets
def transferNET(application,input_size):    
    
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input,GlobalMaxPool2D,Dense,BatchNormalization,Dropout
    
     # Random consistency seed
    tf.random.set_seed(100)
    
    # load application
    tNet = application(input_shape=input_size[1:],include_top=False, weights='imagenet')
    
    # flatten with pooling
    pool = GlobalMaxPool2D(name="CustomLayerStart")(tNet.output)
    
    # classifier branch for car names
    nameBranch = Dense(512,activation='relu')(pool)
    Nbn1 = BatchNormalization()(nameBranch)
    Ndo1 = Dropout(0.3)(Nbn1)
    Nhid1 = Dense(256,activation='relu')(Ndo1)
    Nbn2 = BatchNormalization()(Nhid1)
    Ndo2 = Dropout(0.3)(Nbn2)
    classifier = Dense(196,activation='softmax',name="names")(Ndo2)
    
    # regression branch for bounding boxes
    boxBranch = Dense(64,activation='relu')(pool)
    Bbn1 = BatchNormalization()(boxBranch)
    Bdo1 = Dropout(0.3)(Bbn1)
    Bhid1 = Dense(32,activation='relu')(Bdo1)
    Bbn2 = BatchNormalization()(Bhid1)
    Bdo2 = Dropout(0.3)(Bbn2)
    bBox = Dense(4,activation='relu',name="boxes")(Bdo2)
    
    # assemble the network
    model = Model(inputs=tNet.inputs,outputs=[classifier,bBox])
    
    # freeze application layers and open classifer & regressor for training
    for layer in model.layers[:-15]:
        layer.trainable = False
    
    return model


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploaded',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 0 


st.title('Welcome To Car Model Classifier!')
instructions = """
        upload any image of car from your machine.
        The image you upload will be fed
        through the Deep Neural Network in real-time
        and the output will be displayed to the screen.
        """
st.write(instructions)

from helper import predictor

uploaded_file = st.file_uploader('Upload An Image')
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        display_image = display_image.resize((500,300))
        st.header('This is the image which you selected')
        st.image(display_image)
        prediction, display_image = predictor(os.path.join('uploaded',uploaded_file.name))
        print(prediction)
        os.remove('uploaded/'+uploaded_file.name)
        # drawing graphs
        st.subheader('Predictions :-')

        st.write(prediction)

        #Add bounding boxes on uploaded image
        #display_image = fin_img
        st.image(display_image)



