from RandomGenClass import DataGenerator
import InputOutputForNN as ionn
import pandas as pd
import numpy as np
import cnn_deep_0313 as nn_model
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
import h5py
import pickle
from keras.callbacks import TensorBoard

#Train Test Split parameters
id_num = 'Guo_0315_deep_rd'
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
epochs_number = 20
batch_size = 10
earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min')
mcp_save = ModelCheckpoint('model_'+str(id_num)+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
history = History()
tensorboard = TensorBoard(log_dir='./logs', batch_size=batch_size, histogram_freq=0, write_graph=True, write_images=False)
params ={'dim_x': InputDim[0],
         'dim_y': InputDim[1],
         'dim_z': 3,
         'batch_size': batch_size,
         'shuffle': True}
training_generator = DataGenerator(**params).generate(train_ids,train_df)
validation_generator = DataGenerator(**params).generate(val_ids,train_df)
model.fit_generator(generator=training_generator, steps_per_epoch=len(train_ids)//batch_size, epochs=epochs_number, 
                    validation_data=validation_generator, validation_steps=len(val_ids)//batch_size,
                    callbacks=[earlyStopping,mcp_save,history,tensorboard])
df = pd.DataFrame.from_dict(history.history)
df.to_csv('history_'+str(id_num)+'.csv', sep='\t', index=True, float_format='%.4f')

#Import test data pieces given image type: 'all','fluo','histo' or 'bright'
Test_data = ionn.sub_fragments_extract(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='histo',train=False,reflection=False)
#test data prediction
model.load_weights(filepath = 'model_'+str(id_num)+'.hdf5')
Test_Label = []
for i in range(Test_data.shape[0]):
    test_x = Test_data.loc[i,'X']
    pred_test = model.predict(test_x)
    pred_label = ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
    OutputImage = ionn.OutputStitch(img_shape=Test_data.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
    OutputImage = np.where(OutputImage>cutoff,1,0)
    Test_Label.append((Test_data.loc[i,'ImageId'],OutputImage))
pickle.dump(Test_Label,open( "Test_Label.p","wb" ))
