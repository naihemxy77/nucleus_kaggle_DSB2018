import InputOutputForNN as ionn
import pandas as pd
import numpy as np
import cnn_shallow_0307 as nn_model
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
import h5py
import pickle

#Train Test Split parameters
id_num = 'Guo_0307_shallow'
r = 0.2
SEED = 932894
#Confidence threshold for nuclei identification
cutoff = 0.5

#Fragment parameters
InputDim = [128,128]
OutputDim = [100,100]
Stride = [50,50]
#Import training data pieces given image type: 'all','fluo','histo' or 'bright'
Train_data = ionn.sub_fragments_extract(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='histo',train=True,reflection=False)

#Split images into train val sets (note that pieces for one image always together belong to train/val set)
total_ids = list(Train_data.index.values)
train_ids = random.sample(total_ids,int(len(total_ids)*(1-r)))
val_ids = list(set(total_ids).difference(train_ids))

#concatenate all pieces into one dataset
train_X = np.concatenate(Train_data.loc[train_ids,'X'].values,axis=0)
train_y = np.concatenate(Train_data.loc[train_ids,'y'].values,axis=0)
val_X = np.concatenate(Train_data.loc[val_ids,'X'].values,axis=0)
val_y = np.concatenate(Train_data.loc[val_ids,'y'].values,axis=0)
#model fitting
model = nn_model.model_gen(InputDim)
epochs_number = 10
earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_'+str(id_num)+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
history = History()
model.fit(train_X,train_y, batch_size=32, epochs=epochs_number, verbose=1, validation_data=(val_X,val_y), shuffle=True, class_weight=None, initial_epoch=0, callbacks=[earlyStopping,mcp_save,history])
df = pd.DataFrame.from_dict(history.history)
df.to_csv('history_'+str(id_num)+'.csv', sep='\t', index=True, float_format='%.4f')

#Import test data pieces given image type: 'all','fluo','histo' or 'bright'
Test_data = ionn.sub_fragments_extract(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='histo',train=False,reflection=False)
#test data prediction
model.load_weights(filepath = '.mdl_'+str(id_num)+'.hdf5')
Test_Label = []
for i in range(Test_data.shape[0]):
    test_x = Test_data.loc[i,'X']
    pred_test = model.predict(test_x)
    pred_label = ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
    OutputImage = ionn.OutputStitch(img_shape=Test_data.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
    OutputImage = np.where(OutputImage>cutoff,1,0)
    Test_Label.append((Test_data.loc[i,'ImageId'],OutputImage))
pickle.dump(Test_Label,open( "Test_Label.p","wb" ))