import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Lambda
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
    X = Conv2D(filters=f, kernel_size=(k,k), kernel_initializer='he_normal', padding='same', name=name+'_c1')(X)
    X = BatchNormalization(name=name+'_b1')(X)
    X = Activation('relu', name=name+'_a1')(X)
    X = Dropout(0.1)(X)
    X = Conv2D(filters=f, kernel_size=(k,k), kernel_initializer='he_normal', padding='same', name=name+'_c2')(X)
    X = BatchNormalization(name=name+'_b2')(X)
    X = Activation('relu', name=name+'_a2')(X)
    X = Dropout(0.1)(X)
    X = Conv2D(filters=f, kernel_size=(k,k), kernel_initializer='he_normal', padding='same', name=name+'_c3')(X)
    X = BatchNormalization(name=name+'_b3')(X)
    X = Activation('relu', name=name+'_a3')(X)
    X = Dropout(0.1)(X)
    return X

def F(x0,name):
    x0 = Reshape((128,128,1), name=name+'_re')(x0)
    x1 = cnn_block(x0,3,3,name+'_C1')
    x2 = concatenate([x0,x1], name='concat_'+name+'_1')
    x3 = cnn_block(x2,16,3,name+'_C2')
    x4 = concatenate([x2,x3], name='concat_'+name+'_2')
    x5 = cnn_block(x4,32,30,name+'_C3')
    x1_sum = cnn_block(x1,1,1,name+'_x1_sum')
    x3_sum = cnn_block(x3,1,1,name+'_x3_sum')
    x3_sum = Dropout(0.15)(x3_sum)
    x5_sum = cnn_block(x5,1,1,name+'_x5_sum')
    x5_sum = Dropout(0.15)(x5_sum)
    x_sum = concatenate([x0,x1_sum,x3_sum,x5_sum], name='concat_'+name+'_sum')
    x_out = cnn_block(x_sum,1,1,name+'_out')
    return x_out

def model_gen(InputDim):
    print('Building model ...')
    inputs = Input((InputDim[0],InputDim[1],6))
    rgb_inputs = Lambda(lambda x: x[:,:,:,:3], output_shape=(InputDim[0],InputDim[1],3), name='rgb_band')(inputs)
    rgb_inputs = GlobalAveragePooling2D()(rgb_inputs)
    h_input = Lambda(lambda x: x[:,:,:,3], output_shape=(InputDim[0],InputDim[1],1), name='h_band')(inputs)
    e_input = Lambda(lambda x: x[:,:,:,4], output_shape=(InputDim[0],InputDim[1],1), name='e_band')(inputs)
    d_input = Lambda(lambda x: x[:,:,:,5], output_shape=(InputDim[0],InputDim[1],1), name='d_band')(inputs)
    h_out = F(h_input,'h_band_f')
    e_out = F(e_input,'e_band_f')
    d_out = F(d_input,'d_band_f')
    combined_out = concatenate([h_out,e_out,d_out], name='concat_hed_combined')
    hidden_weight = Dense(10, activation='relu', kernel_initializer='he_normal', use_bias=False, name='hidden_layer')(rgb_inputs)
    rgb_weight = Dense(3, activation='relu', kernel_initializer='he_normal', use_bias=False, name='hsv_layer')(hidden_weight)
    hed_out = multiply([combined_out, rgb_weight])
    hed_out2 = cnn_block(hed_out,10,1,'pray')
    outputs = Conv2D(filters=1, kernel_size=(1,1), kernel_initializer='he_normal', dilation_rate=1, padding='same', activation='sigmoid')(hed_out2)
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=0.0005, decay=0.0)
    model.compile(loss=jaccard_loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    print('Model Construction Finished.')
    return model

model=model_gen([128,128])
model.summary()