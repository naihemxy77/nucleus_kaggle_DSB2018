#Note: if run on Google GPU, you may want to install keras manually before you run the following codes
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras.optimizers import SGD, Adam

import numpy as np
from sklearn.model_selection import KFold

import pandas as pd
import pickle
train_X=pickle.load(open('../data/trainX_histo.p','rb'))
train_y=pickle.load(open('../data/trainy_histo.p','rb'))

train_X.shape #(number_of_samples,128, 128, 3)
train_y.shape #(number_of_samples, 100, 100)
train_Y=np.zeros((train_X.shape[0],train_X.shape[1],train_X.shape[2], 1), dtype=np.bool)
train_Y[0:6048, 0:128, 0:128,0]=train_y


def get_unet(IMG_WIDTH=train_X.shape[1],IMG_HEIGHT=train_X.shape[2],IMG_CHANNELS=3,dropout=dropout,activate=activate,learning_rate=learning_rate):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(16, (3, 3), activation=activate, kernel_initializer='he_normal', padding='same') (inputs)
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
    adam = Adam(lr=learning_rate, decay=0.0)
    model.compile(optimizer=adam,loss='binary_crossentropy')
    return model

#simple test
dropout=0.1
activate='relu'
learning_rate=0.001
model=get_unet(IMG_WIDTH=train_X.shape[1],IMG_HEIGHT=train_X.shape[2],IMG_CHANNELS=3,dropout=dropout,activate=activate,learning_rate=learning_rate)
results = model.fit(train_X, train_Y, batch_size=50,validation_split=0.1, epochs=50)


#cross validation
CV=3
evaluation=np.zeros((CV,2))
kf = KFold(n_splits=CV,random_state=100)
epochs_number=5
batch_size=50
def cross_validation_keras(CV,epochs_number,train_X, train_Y,dropout,activate,batch_size,learning_rate):
    i=-1
    loss=[]
    val_loss=[]
    for train_index, test_index in kf.split(train_X):
        i=i+1
        model=get_unet(IMG_WIDTH=train_X.shape[1],IMG_HEIGHT=train_X.shape[2],IMG_CHANNELS=3,dropout=dropout,activate=activate,learning_rate=learning_rate)
        x_train, x_val = train_X[train_index], train_X[test_index]
        y_train, y_val = train_Y[train_index], train_Y[test_index]
        history = History()
        earlyStopping = EarlyStopping(patience=5, verbose=1)
        history=model.fit(x_train,y_train, batch_size=batch_size,epochs=epochs_number, validation_data=(x_val,y_val), shuffle=True, initial_epoch=0,callbacks=[earlyStopping])
        loss.append(np.mean(np.array(history.history['loss'])))
        val_loss.append(np.mean(np.array(history.history['val_loss'])))
    average_loss=np.mean(np.array(loss))
    average_val_loss=np.mean(np.array(val_loss))
    print(average_loss)
    print(average_val_loss)
    return average_loss,average_val_loss
#loop needed for testing the parameters
#drop_out_pool=[0.1,0.15,0.2]
#activate=['relu','elu']
#lr_pool=[0.00001,0.00005,0.0001,0.0005,0.001]
drop_out_pool=[0.1,0.15,0.2]
activate_pool=['relu','elu']
lr_pool=[0.0005,0.001,0.005]
for dropout in drop_out_pool:
    for activate in activate_pool:
        for learning_rate in lr_pool:
            print("drop_out: ",dropout)
            print("learning_rate: ",learning_rate)
            print("activate_function: ",activate)
            average_loss,average_val_loss=cross_validation_keras(CV,epochs_number,train_X, train_Y,dropout,activate,batch_size,learning_rate)
            final_loss.append(average_loss)
            final_loss.append(average_val_loss)

pickle.dump(final_loss,open('final_loss.p','wb'))
pickle.dump(final_val_loss,open('final_val_loss.p','wb'))
