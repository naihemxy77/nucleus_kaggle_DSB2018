
# coding: utf-8

'''An implementation of the algorithm proposed in arXiv:1803.00232
    DRUNET: A Dilated-Residual U-Net Deep Learning Network to Digitally Stain Optic Nerve Head Tissues in Optical Coherence Tomography Images
    
    Probably you need to modify the initializer in each Conv2d layer. I used constant just for fun. 
    
    Notice that the learning rate change is scheduled with ReduceLROnPlateau, which is included in the variable "callbacks". 
    Remember to pass it to 'callback' when training the model (model.fit(...,callback=callbacks))
    
    Earlystopping and other checkpoint can also be defined and passed to callbacks. See examples here:
    https://www.programcreek.com/python/example/104413/keras.callbacks.ReduceLROnPlateau
'''

import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Activation, ZeroPadding2D, BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.initializers import glorot_uniform,constant,he_uniform,he_normal
from keras.callbacks import ReduceLROnPlateau
from matplotlib.pyplot import imshow
get_ipython().magic('matplotlib inline')

import keras.backend as K
#K.set_image_data_format('channels_last')
#K.set_learning_phase(1)



def standard_block(X, f, d, stage, block):
    """
    Implementation of the standard block as defined in Figure 2

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, defining the number of filters in the CONV layers of the main path
    d -- integer, defining the number of dilation rates in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the standard block, tensor of shape (n_H = n_H_prev-4, n_W = n_W_prev-4, n_C)
    """
    
    # defining name basis
    conv_name_base = 'std' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # First component of main path
    X = Conv2D(filters = f, kernel_size = (3, 3), dilation_rate = (d,d), strides = (1,1), padding = 'same', name = conv_name_base + '2a', kernel_initializer = constant(1))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '1a')(X)
    X = Activation('elu')(X)
    
    # Second component of main path
    X = Conv2D(filters = f, kernel_size = (3, 3), dilation_rate = (d,d), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = constant(1))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '1b')(X)
    X = Activation('elu')(X)
    
    return X



def residual_block(X, f, d, stage, block):
    """
    Implementation of the residual block as defined in Figure 2

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, defining the number of filters in the CONV layers of the main path
    d -- integer, defining the number of dilation rates in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the residual block, tensor of shape n_H = n_H_prev-2*d, n_W = n_W_prev-2*d, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
        
    # Save the input value. 
    X_shortcut = X

    
    # First component of main path
    X = Conv2D(filters = f, kernel_size = (3, 3), dilation_rate=(d, d), strides = (1,1), padding = 'same', name = conv_name_base + '2a', kernel_initializer = constant(1))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('elu')(X)
    
    # Second component of main path
    X = Conv2D(filters = f, kernel_size = (3, 3), dilation_rate = (d, d), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = constant(1))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    
    # Third component of main path 
    X_shortcut = Conv2D(filters = f, kernel_size = (1, 1), dilation_rate = (d, d), strides = (1,1), padding = 'same', name = conv_name_base + '2c', kernel_initializer = constant(1))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
    # Final step: Add shortcut value to main path, and pass it through a ELU activation
    X = Add()([X, X_shortcut])
    X = Activation('elu')(X)
    
    return X



def DRUNET_gen(input_shape, init_lr = 0.1):
    X_input = Input(input_shape)
    X = ZeroPadding2D((0, 0))(X_input)
    
    X_std1 = standard_block(X=X,f=16,d=1,stage=1,block='a')
    
    X_std1_downSampled = MaxPooling2D(strides=(2,2))(X_std1)
    X_res1 = residual_block(X=X_std1_downSampled, f=16, d=2, stage=1,block='b')
    
    X_res1_downSampled = MaxPooling2D(strides=(2,2))(X_res1)
    X_res2 = residual_block(X=X_res1_downSampled, f=16, d=4, stage=1, block='c')
    
    X_res2_downSampled = MaxPooling2D(strides=(2,2))(X_res2)
    X_res3 = residual_block(X=X_res2_downSampled,f=16,d=8,stage=1,block='d')
    
    X_res3_upSampled = UpSampling2D((2,2))(X_res3)
    X_res4 = residual_block(X=Add()([X_res3_upSampled,X_res2]),f=16,d=4,stage=2,block='e') #
    
    X_res4_upSampled = UpSampling2D((2,2))(X_res4)
    X_res5 = residual_block(X=Add()([X_res4_upSampled,X_res1]),f=16,d=2,stage=2,block='f')
    
    X_res5_upSampled = UpSampling2D((2,2))(X_res5)
    X_std2 = standard_block(X=Add()([X_res5_upSampled,X_std1]),f=16,d=1,stage=2,block='g')
    
    X_output = Conv2D(1, (1, 1), activation='sigmoid')(X_std2)
    
    model = Model(inputs=[X_input], outputs=[X_output])
    #adam = Adam(lr = learning_rate)
    sgd = SGD(lr=init_lr, decay=0.0,momentum = 0.9,nesterov=True)
    model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
    return model


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
callbacks = []
callbacks.append(reduce_lr)
#        model.fit(x_train,
#                  y_train,
#                  epochs=5,
#                  callbacks=callbacks,
#                  validation_data=(x_valid, y_valid))




