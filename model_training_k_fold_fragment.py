import InputOutputForNN as ionn
import pandas as pd
import numpy as np
import ZoomNet_0320 as nn_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
import h5py
import pickle
from sklearn.model_selection import KFold

#Train Test Split parameters
n = 5
id_num = 'Guo_0405_Zoom_invert_histo_'+str(n)+'fold'
SEED = 932894
#Confidence threshold for nuclei identification
cutoff = 0.5

#Fragment parameters
InputDim = [128,128]
OutputDim = [100,100]
Stride = [50,50]
#Import training data pieces given image type: 'all','fluo','histo' or 'bright'
Train_data = ionn.sub_fragments_extract(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='histo',train=True,reflection=False)
ImageIds = Train_data.loc[:,'ImageId']

from keras import backend as K
# set GPU memory 
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

#Split images into cross-fold sets (note that pieces for one image always together belong to train/val set)
kf = KFold(n_splits=n, shuffle=True, random_state=SEED)
ids=list(kf.split(ImageIds))
loc_i = np.arange(n)

def model_fitting(ids,I):
    #concatenate all pieces into one dataset
    train_X = np.concatenate(Train_data.loc[ids[0],'X'].values,axis=0)
    train_y = np.concatenate(Train_data.loc[ids[0],'y'].values,axis=0)
    val_X = np.concatenate(Train_data.loc[ids[1],'X'].values,axis=0)
    val_y = np.concatenate(Train_data.loc[ids[1],'y'].values,axis=0)
    #model fitting
    model = nn_model.model_gen(InputDim)
    epochs_number = 100
    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('model_'+str(id_num)+'_'+str(I)+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    history = History()
    output_history = model.fit(train_X,train_y, batch_size=32, epochs=epochs_number, verbose=1, validation_data=(val_X,val_y), shuffle=True, class_weight=None, initial_epoch=0, callbacks=[earlyStopping,mcp_save,history])
    df = pd.DataFrame.from_dict(history.history)
    df.to_csv('history_'+str(id_num)+'_'+str(I)+'.csv', sep='\t', index=True, float_format='%.4f')
    del output_history
    del model

def model_predict(I,Test_data):
    model = nn_model.model_gen(InputDim)
    #test data prediction
    print(str(I)+'th cv model to predict...')
    model.load_weights(filepath = 'model_'+str(id_num)+'_'+str(I)+'.hdf5')
    Test_Label_I = []
    for t in range(Test_data.shape[0]):
        print(str(t)+'th image is being predicted...')
        test_x = Test_data.loc[t,'X']
        pred_test = model.predict(test_x)
        Test_Label_I.append(pred_test)
    ##To release GPU memory by deleting model
    del model
    return Test_Label_I

for i in range(n):
    print(str(i)+'th run is starting...')
    model_fitting(ids[i],i)

#Import test data pieces given image type: 'all','fluo','histo' or 'bright'
Test_data = ionn.sub_fragments_extract(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='all',train=False,reflection=False)
print('Start to predict...')
#pred_outputs_kfold = []
for i in range(n):
    if i == 0:
        pred_outputs_kfold=np.array(model_predict(i,Test_data))
    else:
        pred_outputs_kfold=pred_outputs_kfold+np.array(model_predict(i,Test_data))
print(pred_outputs_kfold.shape)
pred_outputs_kfold = pred_outputs_kfold/n
print('Preparing test labels...')
Test_Label = []
for i in range(Test_data.shape[0]):
    print(str(i)+'th model is processing...')
    pred_test = pred_outputs_kfold[i]
    pred_label = ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
    OutputImage = ionn.OutputStitch(img_shape=Test_data.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
    #OutputImage = np.where(OutputImage>cutoff,1,0)
    Test_Label.append((Test_data.loc[i,'ImageId'],OutputImage))
print('Saving results...')
pickle.dump(Test_Label,open( "Test_Label.p","wb" ))

del Test_data
del Test_Label
del pred_outputs_kfold
#Import test data pieces (rotated) given image type: 'all','fluo','histo' or 'bright'
Test_data_rot = ionn.sub_fragments_extract_rot(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='all',train=False,reflection=False)
print('Start to predict...')
#pred_outputs_kfold = []
for i in range(n):
    if i == 0:
        pred_outputs_kfold=np.array(model_predict(i,Test_data_rot))
    else:
        pred_outputs_kfold=pred_outputs_kfold+np.array(model_predict(i,Test_data_rot))
print(pred_outputs_kfold.shape)
pred_outputs_kfold = pred_outputs_kfold/n
print('Preparing test labels...')
Test_Label_rot = []
for i in range(Test_data_rot.shape[0]):
    print(str(i)+'th model is processing...')
    pred_test = pred_outputs_kfold[i]
    pred_label = ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
    OutputImage = ionn.OutputStitch(img_shape=Test_data_rot.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
    #OutputImage = np.where(OutputImage>cutoff,1,0)
    Test_Label_rot.append((Test_data_rot.loc[i,'ImageId'],OutputImage))
print('Saving results...')
pickle.dump(Test_Label_rot,open( "Test_Label_rot.p","wb" ))