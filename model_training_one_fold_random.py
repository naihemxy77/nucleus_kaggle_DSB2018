from RandomGenClass import DataGenerator
import InputOutputForNN as ionn
import pandas as pd
import numpy as np
import SillyNet as nn_model
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
import h5py
import pickle

#Train Test Split parameters
id_num = 'trial'
r = 0.1
SEED = 932894
#Confidence threshold for nuclei identification
cutoff = 0.5

train_df = pickle.load(open("./inputs/train_df.p","rb"))
random.seed(124335)
#Fragment parameters
InputDim = [128,128]
OutputDim = [100,100]
Stride = [50,50]
#Extract train data imageids
#train_df = train_df
#total_ids = list(train_df['ImageId'].values)
#If just want to train fluorescent data (similarly, 1 for histo and 2 for bright)
train_df = train_df[train_df['hsv_cluster']==1]
total_ids = list(train_df.loc[train_df['hsv_cluster']==1,'ImageId'].values)

#Split images into train val sets
train_ids = random.sample(total_ids,int(len(total_ids)*(1-r)))
val_ids = list(set(total_ids).difference(train_ids))

#model fitting
model = nn_model.model_gen(InputDim)
epochs_number = 100
batch_size = 10
#earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min')
mcp_save = ModelCheckpoint('./trial/model_'+str(id_num)+'.hdf5', save_best_only=True, monitor='val_jaccard_coef', mode='max')
history = History()
params ={'dim_x': InputDim[0],
         'dim_y': InputDim[1],
         'dim_z': 3,
         'batch_size': batch_size,
         'shuffle': True}
training_generator = DataGenerator(**params).generate(train_ids,train_df)
validation_generator = DataGenerator(**params).generate(val_ids,train_df)
model.fit_generator(generator=training_generator, steps_per_epoch=len(train_ids)//batch_size, epochs=epochs_number, 
                    validation_data=validation_generator, validation_steps=len(val_ids)//batch_size,
                    callbacks=[mcp_save,history])
df = pd.DataFrame.from_dict(history.history)
df.to_csv('./trial/history_'+str(id_num)+'.csv', sep='\t', index=True, float_format='%.4f')