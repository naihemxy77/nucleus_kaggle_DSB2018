import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout,BatchNormalization, Activation, Lambda
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras.optimizers import Adam
from keras.layers.merge import concatenate
import keras.backend as K

np.random.seed(534899574)

def reflect_extend(X,InputDim):
    X_hort = Lambda(lambda x: K.reverse(x,axes=2),output_shape=(InputDim[0],InputDim[1],3), name='hort')(X)
    X_vert = Lambda(lambda x: K.reverse(x,axes=1),output_shape=(InputDim[0],InputDim[1],3), name='vert')(X)
    X_diag = Lambda(lambda x: K.reverse(x,axes=1),output_shape=(InputDim[0],InputDim[1],3), name='diag')(X_hort)
    X_1 = concatenate([X_diag,X_vert,X_diag],axis=-2)
    X_2 = concatenate([X_hort,X,X_hort],axis=-2)
    X_e = concatenate([X_1,X_2,X_1],axis=-3)
    X_e = Lambda(lambda x: x[:,122:262,122:262,:], output_shape=(140,140,3), name='crop')(X_e)
    return X_e

def cnn_block(X,f,k,d,name):
    X = Conv2D(filters=f, kernel_size=(k,k), kernel_initializer='he_normal', dilation_rate=d, padding='valid', name=name+'_c')(X)
    X = BatchNormalization(name=name+'_b')(X)
    X = Activation('relu')(X)
    return X

def squeeze_excite_block(x0, ratio=10):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
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
    inputs = Input((InputDim[0],InputDim[1],9))
    inputs_s = Lambda(lambda x: x[:,:,:,:3], output_shape=(InputDim[0],InputDim[1],3), name='short')(inputs)
    inputs_b = Lambda(lambda x: x[:,:,:,3:], output_shape=(InputDim[0],InputDim[1],6), name='bg')(inputs)
    inputs_e = reflect_extend(inputs_s,InputDim)
    #Local
    loc_view = cnn_block(inputs_e,8,3,1,'local')
    inputs_s_l = Lambda(lambda x: x[:,1:139,1:139,:], output_shape=(138,138,3), name='input_loc')(inputs_e)
    loc_view_c = concatenate([loc_view,inputs_s_l])
    loc_view_c = squeeze_excite_block(loc_view_c)
    #Middle
    mid_view = cnn_block(loc_view_c,16,3,2,'middle')
    inputs_s_m = Lambda(lambda x: x[:,3:137,3:137,:], output_shape=(134,134,3), name='input_mid')(inputs_e)
    loc_v_m = Lambda(lambda x: x[:,1:135,1:135,:], output_shape=(134,134,8), name='local_mid')(loc_view)
    mid_view_c = concatenate([mid_view,loc_v_m,inputs_s_m])
    mid_view_c = squeeze_excite_block(mid_view_c)
    #Global
    glo_view = cnn_block(mid_view_c,32,3,3,'global')
    inputs_s_g = Lambda(lambda x: x[:,6:134,6:134,:], output_shape=(128,128,3), name='input_glo')(inputs_e)
    loc_v_g = Lambda(lambda x: x[:,5:133,5:133,:], output_shape=(128,128,8), name='local_glo')(loc_view)
    mid_v_g = Lambda(lambda x: x[:,3:131,3:131,:], output_shape=(128,128,16), name='middle_glo')(mid_view)
    glo_view_c = concatenate([glo_view,mid_v_g,loc_v_g,inputs_s_g,inputs_b])
    glo_view_c = squeeze_excite_block(glo_view_c)
    #Final processing
    outputs = Conv2D(16,(1,1), kernel_initializer='he_normal', padding='valid', name='final')(glo_view_c)
    outputs = Conv2D(1,(1,1), kernel_initializer='he_normal', padding='valid', name='last')(outputs)
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=0.0005, decay=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    print('Model Construction Finished.')
    return model

model=model_gen([128,128])
model.summary()
