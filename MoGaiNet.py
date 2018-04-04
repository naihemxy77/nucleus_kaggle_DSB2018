# coding: utf-8

# In[1]:

import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout,BatchNormalization, Activation, Cropping2D, Add, Dropout
from keras.optimizers import SGD,Adam
from keras.layers.merge import concatenate

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

np.random.seed(534899574)

# In[2]:


def MergeCrop(layerList):
    edgeList = []
    for layer in layerList:
        edgeList.append(int(layer.shape[1]))
        pass
    edgeLength = min(edgeList)
    newLayerList = []
    for layer in layerList:
        newLayerList.append(Cropping2D(cropping=(int(layer.shape[1])-edgeLength)//2)(layer))
        pass
    merge = concatenate(newLayerList)
    return merge


# In[3]:


def AddCrop(layerList):
    edgeList = []
    for layer in layerList:
        edgeList.append(int(layer.shape[1]))
        pass
    edgeLength = min(edgeList)
    newLayerList = []
    for layer in layerList:
        newLayerList.append(Cropping2D(cropping=(int(layer.shape[1])-edgeLength)//2)(layer))
        pass
    add = Add()(newLayerList)
    return add


# In[4]:


def Shrink_Block(X,f,x,y):
    X0 = Conv2D(filters=f,kernel_size=(2,2),kernel_initializer='he_normal', padding='valid')(X)
    X0 = BatchNormalization(axis=3)(X0)
    X0 = Activation('relu')(X0)
    
    X1 = Conv2D(filters=f,kernel_size=(x,y),kernel_initializer='he_normal', padding='valid')(X)
    X1 = BatchNormalization(axis=3)(X1)
    X1 = Activation('relu')(X1)
    X1 = Conv2D(filters=f,kernel_size=(y,x),kernel_initializer='he_normal', padding='valid')(X1)
    X1 = BatchNormalization(axis=3)(X1)
    X1 = Activation('relu')(X1)
    
    X2 = Conv2D(filters=f,kernel_size=(y,x),kernel_initializer='he_normal', padding='valid')(X)
    X2 = BatchNormalization(axis=3)(X2)
    X2 = Activation('relu')(X2)
    X2 = Conv2D(filters=f,kernel_size=(x,y),kernel_initializer='he_normal', padding='valid')(X2)
    X2 = BatchNormalization(axis=3)(X2)
    X2 = Activation('relu')(X2)
    
    X3 = Conv2D(filters=f,kernel_size=(x+y-1,x+y-1),kernel_initializer='he_normal', padding='valid')(X)
    X3 = BatchNormalization(axis=3)(X3)
    X3 = Activation('relu')(X3)
    
    X4 = MergeCrop([X0,X1,X2,X3])
    X5 = Conv2D(filters=f,kernel_size=(1,1),kernel_initializer='he_normal', padding='valid',activation='relu')(X4)
    
    return X5


# In[5]:


def model_gen(input_shape = [128,128,3],f=32,nClass=1,init_lr=0.02):
    X_input = Input(input_shape)
    
    conv1 = Conv2D(filters=f,kernel_size=(1,1),kernel_initializer='he_normal', padding='valid', activation='relu')(X_input)
    conv1_shortcut = conv1
    
    X1 = Shrink_Block(X=conv1,f=f,x=3,y=2)
    X1_shortcut = X1
    
    X2 = Shrink_Block(X=X1,f=f,x=6,y=3)
    X2_shortcut = X2
    
    X3 = Shrink_Block(X=X2,f=f,x=9,y=4)
    X3 = Dropout(0.25)(X3)
    X4 = Shrink_Block(X=X3,f=f,x=9,y=4)
    
    X5 = MergeCrop([X4,X2_shortcut])
    X5 = Shrink_Block(X=X5,f=f,x=12,y=3)
    
    X6 = MergeCrop([X5,X1_shortcut])
    X6 = Shrink_Block(X=X6,f=f,x=15,y=2)
    
    X7 = MergeCrop([conv1_shortcut,X6])
    X_output = Conv2D(filters=nClass,kernel_size=(1,1),activation='sigmoid')(X7)
    
    model = Model(inputs=[X_input], outputs=[X_output])
    adam = Adam(lr=0.0005, decay=0.0)
    #sgd = SGD(lr=init_lr, decay=0.0, momentum = 0.9, nesterov=True)
    model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
    return model
