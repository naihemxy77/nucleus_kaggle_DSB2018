from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K

layer=4

width=128
height=128
dropout=0.1
kernel_size=3
activate='relu'
def get_unet(IMG_WIDTH=width,IMG_HEIGHT=height,IMG_CHANNELS=3,dropout=dropout,activate=activate):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs)
    c1 = Conv2D(16, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(dropout) (c1)
    c1 = Conv2D(16, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(32, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(dropout) (c2)
    c2 = Conv2D(32, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(dropout) (c3)
    c3 = Conv2D(64, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(dropout) (c4)
    c4 = Conv2D(128, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(dropout) (c5)
    c5 = Conv2D(256, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(dropout) (c6)
    c6 = Conv2D(128, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(dropout) (c7)
    c7 = Conv2D(64, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(dropout) (c8)
    c8 = Conv2D(32, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (3,3), strides=(2,2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(dropout) (c9)
    c9 = Conv2D(16, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
  
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[dice_coef])
    return model

#loop needed for testing the parameters