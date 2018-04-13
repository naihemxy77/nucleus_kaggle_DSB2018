from RandomGenClass import DataGenerator
import InputOutputForNN as ionn
import pandas as pd
import numpy as np
import model_unet_compile_iou_loss20180403 as nn_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
import h5py
import pickle
from sklearn.model_selection import KFold
import random

#Train Test Split parameters
n = 8
id_num = 'Mao_0412_unet_stage2_fluo_'+str(n)+'fold'
SEED = 932894
#Confidence threshold for nuclei identification
cutoff = 0.5

train_df = pickle.load(open("./inputs/train_df_new_extra.p","rb"))

################### set GPU memory, no need for Snyder########
from keras import backend as K
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
##############no need for Snyder###############################

random.seed(124335)
#Fragment parameters
InputDim = [128,128]
OutputDim = [100,100]
Stride = [50,50]
#Extract train data imageids
train_df = train_df
total_ids = list(train_df['ImageId'].values)
#If just want to train fluorescent data (similarly, 1 for histo and 2 for bright)
train_df = train_df[train_df['hsv_cluster']==0]
total_ids = list(train_df.loc[train_df['hsv_cluster']==0,'ImageId'].values)

#Split images into cross-fold sets (note that pieces for one image always together belong to train/val set)
kf = KFold(n_splits=n, shuffle=True, random_state=SEED)
ids=list(kf.split(total_ids))
loc_i = np.arange(n)

def model_fitting(ids,I,train_df):
    #concatenate all pieces into one dataset
    train_ids = [total_ids[i] for i in ids[0]]
    val_ids = [total_ids[i] for i in ids[1]]
    #model fitting
    model = nn_model.get_unet(InputDim)
    epochs_number = 300
    batch_size = 32
    image_number_per_batch = 16
    #earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('model_'+str(id_num)+'_'+str(I)+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    history = History()
    params ={'dim_x': InputDim[0],
             'dim_y': InputDim[1],
             'dim_z': 4,
             'batch_size': batch_size,
             'batch_image': image_number_per_batch,
             'shuffle': True}
    training_generator = DataGenerator(**params).generate(train_ids,train_df)
    validation_generator = DataGenerator(**params).generate(val_ids,train_df)
    output_history = model.fit_generator(generator=training_generator, steps_per_epoch=len(train_ids)//image_number_per_batch, epochs=epochs_number, validation_data=validation_generator,validation_steps=len(val_ids)//image_number_per_batch, callbacks=[mcp_save,history])
    print('done.')
    df = pd.DataFrame.from_dict(history.history)
    df.to_csv('history_'+str(id_num)+'_'+str(I)+'.csv', sep='\t', index=True, float_format='%.4f')
    ##To release GPU memory by deleting model
    del output_history
    del model

for i in range(n):
    print(str(i)+'th run is starting...')
    model_fitting(ids[i],i,train_df)
