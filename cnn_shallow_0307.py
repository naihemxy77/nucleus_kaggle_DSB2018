import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout,BatchNormalization
from keras.optimizers import Adam

np.random.seed(256461)

def model_gen(InputDim):
    print('Building model ...')
    inputs = Input((InputDim[0],InputDim[1],3))
    c1 = Conv2D(filters=4, kernel_size=(10,10),padding='same', activation='relu')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.2)(c1)
    c2 = Conv2D(filters=8, kernel_size=(20,20),padding='same', activation='relu')(c1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.2)(c2)
    c3 = Conv2D(filters=1, kernel_size=(20,20),padding='same', activation='relu')(c2)
    outputs = Conv2D(1,(1,1), activation='sigmoid')(c3)
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=0.001, decay=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model