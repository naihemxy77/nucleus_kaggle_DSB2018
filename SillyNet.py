import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout,BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from skimage.morphology import label

np.random.seed(256461)

def custom_loss(y_pred,y_true):
    return K.max(K.abs(y_pred - y_true))

def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def jaccard_loss(y_true, y_pred):
    return 1-jaccard_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    binary_loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)#-K.mean(y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred))
    jac = jaccard_coef(y_true, y_pred)
    combined_loss = binary_loss-K.log(jac)
    return combined_loss



def model_gen(InputDim):
    print('Building model ...')
    inputs = Input((InputDim[0],InputDim[1],3))
    c1 = Conv2D(filters=4, kernel_size=(3,3),padding='same', activation='relu')(inputs)
    c1 = BatchNormalization()(c1)
    c2 = Conv2D(filters=4, kernel_size=(3,3),padding='same', activation='relu')(c1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.2)(c2)
    outputs = Conv2D(1,(1,1), activation='sigmoid')(c1)
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=0.001, decay=0.0)
    model.compile(loss=combined_loss,#'binary_crossentropy'
                  optimizer=adam,
                  metrics=[jaccard_coef])#'accuracy'
    return model
