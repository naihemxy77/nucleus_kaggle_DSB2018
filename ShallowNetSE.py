
# coding: utf-8

# In[7]:


from keras.models import Model
from keras.layers import Input, Conv2D, Dropout,BatchNormalization, Activation, Cropping2D, Add, Dropout
from keras.optimizers import SGD,Adam
from keras.layers.merge import concatenate
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras.regularizers import l1


# In[2]:


def squeeze_excite_block(X, ratio=8):
    init = X
    channel_axis = -1 #if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)
    
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    
    res = multiply([init, se])
    
    return res


# In[3]:


def MergeCrop(layerList):
    edgeList = []
    for layer in layerList:
        edgeList.append(int(layer.shape[1]))
        pass
    edgeLength = min(edgeList)
    newLayerList = []
    for layer in layerList:
        newLayerList.append(Cropping2D(cropping=(int(layer.shape[1])-edgeLength)//2)(layer))
        pass
    merge = concatenate(newLayerList)
    return merge


# In[4]:


def AddCrop(layerList):
    edgeList = []
    for layer in layerList:
        edgeList.append(int(layer.shape[1]))
        pass
    edgeLength = min(edgeList)
    newLayerList = []
    for layer in layerList:
        newLayerList.append(Cropping2D(cropping=(int(layer.shape[1])-edgeLength)//2)(layer))
        pass
    add = Add()(newLayerList)
    return add


# In[9]:


def Shrink_Block(X,f,x,y):
    X0 = Conv2D(filters=f,kernel_size=(2,2),kernel_initializer='he_normal', padding='valid')(X)
    X0 = BatchNormalization(axis=3)(X0)
    X0 = Activation('relu')(X0)
    
    X1 = Conv2D(filters=f,kernel_size=(x,y),kernel_initializer='he_normal', padding='valid')(X)
    X1 = BatchNormalization(axis=3)(X1)
    X1 = Activation('relu')(X1)
    X1 = Conv2D(filters=f,kernel_size=(y,x),kernel_initializer='he_normal', padding='valid')(X1)
    X1 = BatchNormalization(axis=3)(X1)
    X1 = Activation('relu')(X1)
    
    X2 = Conv2D(filters=f,kernel_size=(y,x),kernel_initializer='he_normal', padding='valid')(X)
    X2 = BatchNormalization(axis=3)(X2)
    X2 = Activation('relu')(X2)
    X2 = Conv2D(filters=f,kernel_size=(x,y),kernel_initializer='he_normal', padding='valid')(X2)
    X2 = BatchNormalization(axis=3)(X2)
    X2 = Activation('relu')(X2)
    
    X3 = Conv2D(filters=f,kernel_size=(x+y-1,x+y-1),kernel_initializer='he_normal', padding='valid')(X)
    X3 = BatchNormalization(axis=3)(X3)
    X3 = Activation('relu')(X3)
    
    X4 = MergeCrop([X0,X1,X2,X3])
    X5 = Conv2D(filters=f,kernel_size=(1,1),kernel_initializer='he_normal', padding='valid',activation='relu')(X4)
    
    X6 = squeeze_excite_block(X5)
    
    return X6


# In[10]:


def model_gen(input_shape = [128,128,4],f=16,nClass=1,init_lr=0.02):
    X_input = Input(input_shape)
    
    conv1 = Conv2D(filters=f,kernel_size=(2,2),kernel_initializer='he_normal', padding='valid', activation='relu')(X_input)
    conv1_shortcut = conv1
    
    X1 = Shrink_Block(X=conv1,f=f,x=3,y=2) # small
    X1_shortcut = X1
    
    X2 = Shrink_Block(X=X1,f=f,x=9,y=4)  # middle2
    X2_shortcut = X2
    
    X3 = Shrink_Block(X=X_input,f=f,x=9,y=4) # middle1
    X3_shortcut = X3
    
    X4 = Shrink_Block(X=conv1,f=f,x=49,y=10) # large1
    #print("large1: ",X4.shape)
    
    X5 = MergeCrop([X2,X3]) # odd and even problem
    X5 = Dropout(0.25)(X5)
    
    X6 = Shrink_Block(X=X5,f=f,x=23,y=22) # middle3
    #print("middle3: ",X6.shape)
    
    X7 = Shrink_Block(X=X2,f=f,x=23,y=22) # large2
    #print("large2: ",X7.shape)
    
    X8 = MergeCrop([X_input,X4,X6,X7])
    X8 = Dropout(0.25)(X8)
    
    X9 = Conv2D(filters=f,kernel_size=(3,3),kernel_initializer='he_normal', padding='valid', activation='relu')(X8)
    X_output = Conv2D(filters=nClass,kernel_size=(1,1),activation='sigmoid')(X9)
    
    model = Model(inputs=[X_input], outputs=[X_output])
    adam = Adam(lr=0.0005, decay=0.0, amsgrad=True)
    #sgd = SGD(lr=init_lr, decay=0.0, momentum = 0.9, nesterov=True)
    model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
    return model
