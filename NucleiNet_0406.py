import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Lambda, Permute
from keras.layers import Reshape, Dense, multiply, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.layers.merge import concatenate
import keras.backend as K

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


def cnn_block(X,f,k,name):
    X = Conv2D(filters=f, kernel_size=(k,k), kernel_initializer='he_normal', padding='same', name=name+'_c')(X)
    X = BatchNormalization(name=name+'_b')(X)
    X = Activation('relu', name=name+'_a')(X)
    return X

def squeeze_excite_block(x0, ratio=4):
    init = x0
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)
    x = multiply([init, se])
    return x

def model_gen(InputDim):
    print('Building model ...')
    inputs = Input((InputDim[0],InputDim[1],3))
    neighbors = cnn_block(inputs,16,3,'neighbors')
    locals1 = cnn_block(neighbors,32,3,'localss')
    locals2 = Dropout(0.15)(locals1)
    size1 = cnn_block(locals2,4,3,'size1')
    size2 = cnn_block(locals2,4,6,'size2')
    size3 = cnn_block(locals2,4,12,'size3')
    size4 = cnn_block(locals2,4,24,'size4')
    size5 = cnn_block(locals2,4,48,'size5')
    size6 = cnn_block(locals2,4,96,'size6')
    nuclei_detect = concatenate([size1,size2,size3,size4,size5,size6], name='concat_sizes')
    nuclei_detect = squeeze_excite_block(nuclei_detect)
    neighbor = cnn_block(neighbors,1,1,'neighbor_summary')
    local = cnn_block(locals1,1,1,'local_summary')
    nuclei = cnn_block(nuclei_detect,1,1,'nuclei_summary')
    final_concat = concatenate([inputs,neighbor,local,nuclei], name='final_concat')
    outputs = Conv2D(filters=1, kernel_size=(1,1), kernel_initializer='he_normal', padding='same', activation='sigmoid')(final_concat)
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=0.0005, decay=0.0)
    model.compile(loss=jaccard_loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    print('Model Construction Finished.')
    return model

model=model_gen([128,128])
model.summary()