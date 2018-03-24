import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout,BatchNormalization, Activation, Lambda
from keras.optimizers import Adam
from keras.layers.merge import concatenate

np.random.seed(534899574)

def cnn_block(X,f,k,name):
    X = Conv2D(filters=f, kernel_size=(k,k), kernel_initializer='he_normal', padding='valid', name=name+'_c', activation='relu')(X)
    X = BatchNormalization(name=name+'_b')(X)
    X = Activation('relu')(X)
    return X

def model_gen(InputDim):
    print('Building model ...')
    inputs = Input((InputDim[0],InputDim[1],3))
    extended_inputs = concatenate([inputs,inputs,inputs],axis=-2)
    extended_inputs = concatenate([extended_inputs,extended_inputs,extended_inputs],axis=-3)
    #local scan
    local1 = cnn_block(extended_inputs, 16, 3, 'local1')
    local2 = cnn_block(local1, 16, 3, 'local2')
    local3 = cnn_block(local2, 16, 3, 'local3')
    local1_o = Lambda(lambda x: x[2:380,2:380,:],output_shape=(378,378,16))(local1)
    local2_o = Lambda(lambda x: x[1:379,1:379,:],output_shape=(378,378,16))(local2)
    mid1 = concatenate([local1_o,local2_o,local3])
    #global scan
    global1 = cnn_block(mid1, 32, 32, 'global1')
    global1 = Dropout(0.2)(global1)
    global2 = cnn_block(global1, 32, 32, 'global2')
    global2 = Dropout(0.2)(global2)
    global3 = cnn_block(global2, 32, 32, 'global3')
    global3 = Dropout(0.2)(global3)
    global4 = cnn_block(global3, 64, 32, 'global4')
    global4 = Dropout(0.2)(global4)
    #local view
    mid_o = cnn_block(local3, 3, 1, 'mid_o')
    local_view = Lambda(lambda x: x[125:253,125:253,:],output_shape=(128,128,3))(mid_o)
    #global view
    global_o = cnn_block(global4, 3, 1, 'global_o')
    global_view = Lambda(lambda x: x[63:191,63:191,:],output_shape=(128,128,3))(global_o)
    #combined different zooming view together
    combined = concatenate([inputs,local_view,global_view])
    combined2 = Conv2D(16,(1,1), kernel_initializer='he_normal', activation='relu', name='combined2')(combined)
    outputs = Conv2D(1,(1,1), kernel_initializer='he_normal', activation='sigmoid', name='outputs')(combined2)
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=0.0005, decay=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    print('Model Construction Finished.')
    return model