#Note: if run on Google GPU, you may want to install keras manually before you run the following codes
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam

def get_unet(InputDim):
    main_input = Input((InputDim[0],InputDim[1],3))

    c1 = Conv2D(16, (5,5), activation='relu', kernel_initializer='he_normal', padding='same') (main_input)
    c1 = Dropout(0.15) (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(32, (5,5), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.15) (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (5,5), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.15) (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (5,5), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.15) (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(256, (5,5), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.15) (c5)
    p5 = MaxPooling2D(pool_size=(2, 2)) (c5)
    
    c6 = Conv2D(512, (5,5), activation='relu', kernel_initializer='he_normal', padding='same') (p5)
    c6 = Dropout(0.15) (c6)

    u7 = Conv2DTranspose(256, (5,5), strides=(2,2), padding='same') (c6)
    u7 = concatenate([u7, c5])
    c7 = Conv2D(256, (5,5), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.15) (c7)
    
    u8 = Conv2DTranspose(128, (5,5), strides=(2,2), padding='same') (c7)
    u8 = concatenate([u8, c4])
    c8 = Conv2D(128, (5,5), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.15) (c8)

    u9 = Conv2DTranspose(64, (5,5), strides=(2,2), padding='same') (c8)
    u9 = concatenate([u9, c3])
    c9 = Conv2D(64, (5,5), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.15) (c9)

    u10 = Conv2DTranspose(32, (5,5), strides=(2,2), padding='same') (c9)
    u10 = concatenate([u10, c2])
    c10 = Conv2D(32, (5,5), activation='relu', kernel_initializer='he_normal', padding='same') (u10)
    c10 = Dropout(0.15) (c10)
    
    # A LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    #lstm_out = LSTM(32)(c10)
    
    #add a new input, in this case, is the outline
    bound_input = Input(shape=(InputDim[0],InputDim[1],1), name='bound_input')
    
    u11 = Conv2DTranspose(16, (5,5), strides=(2,2), padding='same') (c10)
    u11 = concatenate([u11, c1,auxiliary_input], axis=3)
    c11 = Conv2D(16, (5,5), activation='relu', kernel_initializer='he_normal', padding='same') (u11)
    c11 = Dropout(0.15) (c11)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c11)

    model = Model(inputs=[main_input,bound_input], outputs=[outputs])
    adam = Adam(lr=0.0005, decay=0.0)
    model.compile(optimizer=adam,loss='binary_crossentropy')
    return model

#simple test
#dropout=0.1
#activate='relu'
#learning_rate=0.001
#model=get_unet(IMG_WIDTH=train_X.shape[1],IMG_HEIGHT=train_X.shape[2],IMG_CHANNELS=3,dropout=dropout,activate=activate,learning_rate=learning_rate)
#results = model.fit(train_X, train_Y, batch_size=50,validation_split=0.1, epochs=50)


