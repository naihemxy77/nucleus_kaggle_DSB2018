import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout,BatchNormalization, Activation
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras import backend as K
import tensorflow as tf

np.random.seed(534899574)

def iou(y_pred,y_true):
    logits=tf.reshape(y_pred, [-1])
    trn_labels=tf.reshape(y_true, [-1])
    inter=tf.reduce_sum(tf.multiply(logits,trn_labels))
    union=tf.reduce_sum(tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels)))
    loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter,union))
    return loss

def cnn_block(X,f,k):
    X = Conv2D(filters=f, kernel_size=(k,k), kernel_initializer='he_normal', padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    return X

def model_gen(InputDim):
    print('Building model ...')
    inputs = Input((InputDim[0],InputDim[1],3))
    local1 = cnn_block(inputs, 16, 3)
    local2 = cnn_block(local1, 16, 10)
    local3 = concatenate([local1,local2])
    global1 = cnn_block(local3, 32, 30)
    global1 = Dropout(0.2)(global1)
    global2 = cnn_block(global1, 32, 30)
    global2 = Dropout(0.2)(global2)
    global3 = cnn_block(global2, 32, 30)
    global3 = Dropout(0.2)(global3)
    global4 = cnn_block(global3, 64, 30)
    global4 = Dropout(0.2)(global4)
    
    local4 = cnn_block(local3, 3, 1)
    global5 = cnn_block(global4, 3, 1)
    combined = concatenate([inputs,local4,global5])
    outputs = Conv2D(1,(1,1), kernel_initializer='he_normal', activation='sigmoid')(combined)
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=0.0005, decay=0.0, amsgrad=True)
    model.compile(loss=iou,
                  optimizer=adam,
                  metrics=['accuracy'])
    print('Model Construction Finished.')
    return model
