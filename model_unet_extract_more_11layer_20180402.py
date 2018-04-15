#Note: if run on Google GPU, you may want to install keras manually before you run the following codes
from keras.models import Model
from keras.layers import Input, merge, Reshape
from keras.layers.core import Lambda
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
#def antirectifier(x):
#    return x[:,:,0:3]
#
#def antirectifier2(x):
#    return x[:,:,3]
#
##final_dim as (,,3)for the major input
#def antirectifier_output_shape(input_shape):
#    shape = list(input_shape)
#    shape[-1] = 3
#    return tuple(shape)
#
##final_dim as (,,1)for the major input
#def antirectifier_output_shape2(input_shape):
#    shape = list(input_shape)
#    shape[-1] = 1
#    return tuple(shape)
#
def iou(y_pred,y_true):
    logits=tf.reshape(y_pred, [-1])
    trn_labels=tf.reshape(y_true, [-1])
    smooth = tf.constant(1e-12, dtype=tf.float32)
    inter=tf.reduce_sum(tf.multiply(logits,trn_labels))
    union=tf.reduce_sum(tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels)))
    inter_smooth = tf.add(inter,smooth)
    union_smooth = tf.add(union,smooth)
    loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter_smooth,union_smooth))
    return loss

def get_unet(InputDim):
    inputs = Input((InputDim[0],InputDim[1],7))
    main_input = Lambda(lambda x: x[:,:,:,0:3],output_shape=(InputDim[0],InputDim[1],3))(inputs)
    
    c1 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same') (main_input)
    c1 = Dropout(0.15) (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.15) (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.15) (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.15) (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.15) (c5)
    p5 = MaxPooling2D(pool_size=(2, 2)) (c5)
    
    c6 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same') (p5)
    c6 = Dropout(0.15) (c6)

#adding the background layer as an additional input (bg_input1 to 4)
    #bg_input1 = Lambda(lambda x: x[:,0:8,0:8,3:6],output_shape=(8,8,3))(inputs)
    #bg_input1 = Reshape((8,8,3))(bg_input1)
    u7 = Conv2DTranspose(256, (3,3), strides=(2,2), padding='same') (c6)
    u7 = concatenate([u7, c5])
    c7 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.15) (c7)
    
    #bg_input2 = Lambda(lambda x: x[:,0:16,0:16,3:6],output_shape=(16,16,3))(inputs)
    #bg_input2 = Reshape((16,16,3))(bg_input2)
    u8 = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same') (c7)
    u8 = concatenate([u8, c4])
    c8 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.15) (c8)
    
    #bg_input3 = Lambda(lambda x: x[:,0:32,0:32,3:6],output_shape=(32,32,3))(inputs)
    #bg_input3 = Reshape((32,32,3))(bg_input3)
    u9 = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same') (c8)
    u9 = concatenate([u9, c3])
    c9 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.15) (c9)
    
    bg_input4 = Lambda(lambda x: x[:,0:64,0:64,3:6],output_shape=(64,64,3))(inputs)
    bg_input4 = Reshape((64,64,3))(bg_input4)
    u10 = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same') (c9)
    u10 = concatenate([u10, c2,bg_input4])
    c10 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same') (u10)
    c10 = Dropout(0.15) (c10)
    
    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    #lstm_out = LSTM(32)(c10)
    
    #add a new input, in this case, is the outline
    bound_input = Lambda(lambda x: x[:,:,:,6],output_shape=(InputDim[0],InputDim[1],1))(inputs)
    bound_input = Reshape((InputDim[0],InputDim[1],1))(bound_input)
    u11 = Conv2DTranspose(16, (3,3), strides=(2,2), padding='same') (c10)
    #u11 = merge([u11, c1_rest, a_matrix_now], mode='concat',concat_axis=3)
    u11 = concatenate([u11, c1, bound_input])
    c11 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same') (u11)
    c11 = Dropout(0.15) (c11)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c11)

    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=0.0005, decay=0.0)
    model.compile(loss=iou,
                  optimizer=adam,
                  metrics=['accuracy'])
    return model

#simple test
#dropout=0.1
#activate='relu'
#learning_rate=0.001
#model=get_unet(IMG_WIDTH=train_X.shape[1],IMG_HEIGHT=train_X.shape[2],IMG_CHANNELS=3,dropout=dropout,activate=activate,learning_rate=learning_rate)
#results = model.fit(train_X, train_Y, batch_size=50,validation_split=0.1, epochs=50)


