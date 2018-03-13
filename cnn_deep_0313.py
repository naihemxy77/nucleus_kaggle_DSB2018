import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.layers.merge import concatenate

np.random.seed(256461)

def model_gen(InputDim):
    print('Building model ...')
    inputs = Input((InputDim[0],InputDim[1],3))
    c1 = Conv2D(filters=16, kernel_size=(10,10),padding='same', activation='relu')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.2)(c1)
    c2 = Conv2D(filters=32, kernel_size=(20,20),padding='same', activation='relu')(c1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.2)(c2)
    c3 = Conv2D(filters=32, kernel_size=(20,20),padding='same', activation='relu')(c2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c4 = Conv2D(filters=64, kernel_size=(30,30),padding='same', activation='relu')(c3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c5 = Conv2D(filters=64, kernel_size=(30,30),padding='same', activation='relu')(c4)
    c6 = concatenate([c5,c1])
    outputs = Conv2D(1,(1,1), activation='sigmoid')(c6)
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=0.001, decay=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model