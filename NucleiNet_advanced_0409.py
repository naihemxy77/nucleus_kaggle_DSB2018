import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Lambda, Permute, Flatten
from keras.layers import Reshape, Dense, multiply, Dropout
from keras.optimizers import Adam
from keras.layers.merge import concatenate
import keras.backend as K
import tensorflow as tf

np.random.seed(234374632)

def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def jaccard_loss(y_true, y_pred):
    return 1-jaccard_coef(y_true, y_pred)

def iou(y_pred,y_true):
    logits=tf.reshape(y_pred, [-1])
    trn_labels=tf.reshape(y_true, [-1])
    inter=tf.reduce_sum(tf.multiply(logits,trn_labels))
    union=tf.reduce_sum(tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels)))
    loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter,union))
    return loss

def cnn_block(X,f,k,name):
    X = Conv2D(filters=f, kernel_size=(k,k), kernel_initializer='he_normal', padding='same', name=name+'_c')(X)
    X = BatchNormalization(name=name+'_b')(X)
    X = Activation('relu', name=name+'_a')(X)
    return X
  
def screen_block(X,f,k,name):
    X = Conv2D(filters=f*2, kernel_size=(k,k), kernel_initializer='he_normal', padding='same', name=name+'_c')(X)
    X = BatchNormalization(name=name+'_b')(X)
    X = Activation('relu', name=name+'_a')(X)
    X = Conv2D(filters=f, kernel_size=(k,k), kernel_initializer='he_normal', padding='same', name=name+'_c2')(X)
    X = BatchNormalization(name=name+'_b2')(X)
    X = Activation('relu', name=name+'_a2')(X)
    return X

def output_block(X,f,k,name):
    X = Conv2D(filters=f, kernel_size=(k,k), kernel_initializer='he_normal', padding='same', name=name+'_c')(X)
    X = BatchNormalization(name=name+'_b')(X)
    X = Activation('sigmoid', name=name+'_a')(X)
    return X

def standard_dev(x):
    flat= Flatten()(x)
    std = Lambda(lambda x: K.std(x, axis=1))(flat)
    return std

def layer_stat_cal(x):
    flat= Flatten()(x)
    avg = Lambda(lambda x: K.mean(x, axis=1))(flat)
    avg = Reshape((1,1))(avg)
    std = Lambda(lambda x: K.std(x, axis=1))(flat)
    std = Reshape((1,1))(std)
    stat = concatenate([avg,std], axis=2)
    return stat

def squeeze_excite_block(x0,se,ratio=4):
    init = x0
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters // ratio, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = Reshape((1,filters//ratio,1))(se)
    se = concatenate([se,se,se,se],axis=-1)
    se = Reshape((1,1,filters))(se)
    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)
    x = multiply([x0, se])
    return x

def model_gen(InputDim):
    print('Building model ...')
    inputs = Input((InputDim[0],InputDim[1],3))
    neighbors = cnn_block(inputs,16,3,'neighbors')
    locals1 = cnn_block(neighbors,32,3,'localss')
    locals2 = Dropout(0.15)(locals1)
    size1 = screen_block(locals2,4,3,'size1')
    stat1 = layer_stat_cal(size1)
    size2 = screen_block(locals2,4,6,'size2')
    stat2 = layer_stat_cal(size2)
    size3 = screen_block(locals2,4,12,'size3')
    stat3 = layer_stat_cal(size3)
    size4 = screen_block(locals2,4,24,'size4')
    stat4 = layer_stat_cal(size4)
    size5 = screen_block(locals2,4,48,'size5')
    stat5 = layer_stat_cal(size5)
    size6 = screen_block(locals2,4,96,'size6')
    stat6 = layer_stat_cal(size6)
    nuclei_detect= concatenate([size1,size2,size3,size4,size5,size6], name='concat_sizes')
    nuclei_stat = concatenate([stat1,stat2,stat3,stat4,stat5,stat6], axis=-1,name='concat_stat')
    nuclei_detect = squeeze_excite_block(nuclei_detect,nuclei_stat)
    neighbor = output_block(neighbors,1,1,'neighbor_summary')
    local = output_block(locals1,1,1,'local_summary')
    nuclei = output_block(nuclei_detect,1,1,'nuclei_summary')
    final_concat = concatenate([inputs,neighbor,local,nuclei], name='final_concat')
    outputs = Conv2D(filters=1, kernel_size=(1,1), kernel_initializer='he_normal', padding='same', activation='sigmoid')(final_concat)
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=0.0005, decay=0.0)
    model.compile(loss=iou,
                  optimizer=adam,
                  metrics=['accuracy'])
    print('Model Construction Finished.')
    return model

#model=model_gen([128,128])
#model.summary()